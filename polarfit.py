#functions
#for actual scripts, see polarfit_scripts
#Analysis of daily temperature logs from weather stations and sea ice extent logs
#data extraction, analysis and plotting

#uses the following folder structure:
#place weather source data in "source_data_weather_stations"  folder
#place ice area source data in "source_ice" folder
#create the following folders for intermediare data:
#"p2fit", "r2 2yr", "r2 year", "r2 half", "r2 mth", "r2 raw"
#output data folders are generated automatically

import os
import sys
import numpy as np
from math import *
import random
import matplotlib.pyplot as plt
import pickle
import time
from scipy import optimize
import requests
from bs4 import BeautifulSoup
from numba import jit, njit

#WMO numbers of polar meteo stations
Blashyrkh = [20107, 20087, 20069, 20046, 20292, 21432, 20891, 20674, 24266, 24688, 25051, 25282, 21982, 26063, 22113, 27612, 21824, 23226, 24959, 25563, 21946, 25248, 'ice']
names = ['Barentzburg', 'Golomyanny', 'Wiese', 'Hayes', 'Chelyuskin', 'Kotelny', 'Khatanga', 'Dixon', 'Verkhoyansk', 'Oymyakon', 'Pevek', 'Vankarem', 'Wrangel', 'SPB', 'Murmansk', 'Default City', 'Tiksi', 'Vorkuta', 'Yakutsk', 'Anadyr', 'Chokurdah', 'Ilirney', 'Ice']
m_lengths = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
m_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
sm_types = ['2yr', 'year', 'half', 'mth', 'raw'] #half-widths of smoothing windows used in the following data processing


#===== calendar

#@jit(nopython=True)
def isleap(yr): #OK #https://astroconverter.com/utcmjd.html to check
 is_leap = False
 nlc = (100, 200, 300)
 if yr % 4 == 0 and yr % 400 not in nlc:
  is_leap = True
 return is_leap

#conversion between Modified Julian Day and year float value
#@jit(nopython=True)
def mjd_to_yr_s(d):
 yr = d/365.25 + 1858.87
 return yr

#@jit(nopython=True)
def mjd_to_yr(t_list):
 t_list_1 = t_list
 for i in range(len(t_list)):
  yr = t_list[i][0]/365.25 + 1858.87
  t_list_1[i].insert(1, yr)
 return t_list_1 

#conversion of month-and-day values to day number (1 - 366) 
#@jit(nopython=True)
def md_to_daynum(m, d):
 return d + sum(m_lengths[:m-1])  

#===== math simple

#average of a list
#@jit(nopython=True)
def avg(lst_1):
 lst = tuple(lst_1)
 return sum(lst)/len(lst)

#returns (root square deviation, average) of a list
#@jit(nopython=True)
def rsd(val_list): 
 sum_sqr = 0
 sigma = 0
 avg = sum(val_list) / len(val_list)
 for i in range(len(val_list)):
  sum_sqr += (val_list[i] - avg)**2
 sigma = sqrt(sum_sqr/len(val_list))
 return sigma, avg
 
#calculates sum of squared deviations between data and linear model y = k*x+b
#@jit(nopython=True)
def sq_res(data_in, fit):
 ord_0 = fit[0][1] #b
 ord_1 = fit[0][0] #k
 sq_sum = 0
 for i in range(len(data_in)):
  calc_val = ord_1 * data_in[i][0] + ord_0
  res = data_in[i][1] - calc_val
  sq_sum += res ** 2
 return sq_sum

#temperature to rgb conversion for visualizations
def t_to_rgb(t_in):
 r = g = b = 0
 t = t_in / 100
 if t < 10:
  t = 10
 elif t > 4000:
  t = 400
 if t < 66: # red channel
  r = 255
 else:
  r = 329.698727446 * ((t-60)**-0.1332)
 if r < 0:
  r = 0
 elif r > 255:
  r = 255
 if t <= 66: #green channel
  g = 99.4708025861 * log(t) - 161.1195681661
 else:
  g = 288.1221695283 * (t - 60)**-0.0755148492
 if g < 0:
  g = 0
 elif g > 255:
  g = 255
 if t >= 66: #blue channel
  b = 255
 elif t <= 19:
  b = 0
 else:
  b = 138.5177312231 * log(t - 10) - 305.0447927307
 if b < 0:
  b = 0
 elif b > 255:
  b = 255
 r_hex = hex(int(r))[2:].zfill(2)
 g_hex = hex(int(g))[2:].zfill(2)
 b_hex = hex(int(b))[2:].zfill(2)
 return r_hex + g_hex + b_hex

#calculates determination coefficient (R-squared simple and adjusted) for data and model
# takes piece-wise model in form [[x_start_1, slope_1, intercept_1, x_end_1], [x_start_2, ... ]]
#as returned by fitting functions (see below)
#returns also calculated values according to model
#@jit(nopython=True)
def r_sqr(data_in, model):
 simple_avg = avg([d[1] for d in data_in])
 splines = len(model)
 spline = 0
 k = model[spline][1]
 b = model[spline][2]
 sqrs_fit = sqrs_total = 0
 data_calc = []
 for i in range(len(data_in)):
  date_cur = data_in[i][0]
  if date_cur > model[spline][3] and not (spline >= splines - 1):
   spline += 1
   k = model[spline][1]
   b = model[spline][2] 
  val_cur = data_in[i][1]
  val_calc = k * date_cur + b
  data_calc.append([date_cur, val_calc])
  res = val_cur - val_calc
  res_total = val_cur - simple_avg
  sqrs_fit += res ** 2  
  sqrs_total += res_total ** 2
 det_coeff = 1 - sqrs_fit / sqrs_total
 n_pts = len(data_in)
 n_df = 1 + len(model) 
 det_coeff_adj = 1 - (1 - det_coeff**2) * (n_pts - 1) / (n_pts - n_df)
 return det_coeff, det_coeff_adj, data_calc


#=====data extraction
 
#basic_site = 'http://pogodaiklimat.ru/monitor.php?id=' + '27612' + '&month=2&year=2022' 
#weather_page = requests.get(basic_site, verify=False).text
#https://medium.com/geekculture/web-scraping-tables-in-python-using-beautiful-soup-8bbc31c5803e
#extracts table of temperatures from a single month of a single site
def weather_tbl(wpage): 
 weather_page = requests.get(wpage, verify=False).text
 soup = BeautifulSoup(weather_page, 'html.parser')
 wt_site = soup.find("div", {"class": "climate-table-wrap"}) #<div class="climate-table-wrap">
 rows = wt_site.find_all('tr')
 weather_table = []
 empty_table = True
 for d in range(2, len(rows)):
  columns = rows[d].find_all('td')
  row_cur = []
  for i in range(len(columns)):
   val = columns[i].text
   row_cur.append(float(columns[i].text) if val != '' else 999.0)
   row_cur[0] = int(row_cur[0])
  weather_table.append(row_cur)
 return weather_table #day, tmin, tavg, tmax, tanom, precip

  
#builds continuous weather record from a single site by weather_tbl function
#returns year, month, day, t_min, t_avg, t_max, t_anom, precipitation
#writes .csv
#t_anoms are recalculated in all data processing

def weather_extract(site): 
 wpage = 'http://pogodaiklimat.ru/monitor.php?id=' + str(site) + '&month=1&year=2023'
 weather_page = requests.get(wpage, verify=False).text
 soup = BeautifulSoup(weather_page, 'html.parser')
 t_span = soup.find('ul', {'class': 'climate-list'})
 paragr = t_span.find_all('li')
 allspans = paragr[len(paragr)-1].find_all('span') 
 yr_start = int(allspans[len(allspans)-1].text[:4])
 print(yr_start)	
 mth_cur = (yr_start) * 12 + 1 
 tbl_all = []
 t_0 = time.time()
 while mth_cur < 24277: # 2023 Feb
  time.sleep(random.uniform(0.1, 0.9))
  mth_cur += 1
  mth = mth_cur % 12
  yr = mth_cur // 12 - (1 if mth == 0 else 0)
  mth = 12 if mth == 0 else mth
  weather_site = 'http://pogodaiklimat.ru/monitor.php?id=' + str(site) + '&month=' + str(mth) + '&year=' + str(yr)
  print('%.2f' % (time.time() - t_0), yr, mth)
  w_cur = weather_tbl(weather_site)
  for i in range(len(w_cur)):
   w_cur[i].insert(0, mth)
   w_cur[i].insert(0, yr)
  tbl_all += w_cur
 csv_name = str(site) + '.csv' 
 os.chdir('source_data_weather_stations')
 with open (csv_name, 'w') as tbl: #creates log txt with and new names
  for i in range(len(tbl_all)):
   tbl_str = ''
   for j in range(len(tbl_all[i])):
    val = 'ND' if tbl_all[i][j] == 999.0 else str(tbl_all[i][j])#.replace('.', ',') #for xls
    tbl_str += val + '\t'
   tbl.write(tbl_str + '\n')
 os.chdir('..')
 return tbl_all
  
#===== data prepare

#reads csv logs written by weather_tbl  
#temps only! 
#deletes erratic data from extracted logs
#extracts min, avg, max temps and precipitation
def csvread(filename):
 tbl = []
 with open(filename, 'r') as csv:
  lines = csv.readlines()
  for i in range(len(lines)):
   newstr = lines[i].split('\t')
   line = []
   for j in range(len(newstr)-1):
    elt = newstr[j]
    if elt != 'ND':
     if j < 3:
      elt = int(elt) #date
     else:
      elt = float(elt) #values
    line.append(elt)
   if (not isleap(line[0])) and line[1] == 2 and line[2] == 29:
    continue
   else:
    tbl.append(line)
 return tbl   

#adds MJD day numeration
def daynum(tbl): #OK
 tbllen = len(tbl)
 for i in range(tbllen):
  tbl[tbllen - 1 - i].insert(0, 59975-i) #59975 - MJD 2023.01.31
 return tbl

#deletes line if some of the data is missing
def nd_del(tbl): #OK 
 tbl_out = []
 for i in range(len(tbl)):
  isvalid = True
  line = tbl[i]
  if line[4] == 'ND' or line[5] == 'ND' or line[6] == 'ND' or line[7] == 'ND':
   isvalid = False
   continue
  if line[len(line)-1] == 'ND':
   line[len(line)-1] = 0.0 #deletes lines with prec data
  if abs(line[6] - line[4]) < 0.5 and line[5] == line[4] or line[5] == line[6]:
   isvalid = False #deletes lines with wrong temps
  if isvalid:
   tbl_out.append(line)
 return tbl_out

#performs all data preparation on raw temperature data csv
def data_prepare(filename): #OK 
 os.chdir('source_data_weather_stations')
 tbl = csvread(filename) #ice separately! #temp logs have more columns and need cleaning 
 os.chdir('..')
 tbl = daynum(tbl)
 tbl = nd_del(tbl)
 return tbl

#add ice read and prepare here

#===== data process

#calculates running window averaged values for simple XY table data
#calculated window centered on current data point, by x-value and half-width
#while iterating over data points, adds and deletes data points from current window by x-vals
#@jit(nopython=True)
def smoother(tbl_in, hw, deg): 
 halfwidth = hw #hw by x-coord, mind units!
 #tbl_in = list(zip(xvals, yvals))
 smoothie = []
 len_all = len(tbl_in)
 x_cur = tbl_in[0][0]
 window_x = [d[0] for d in tbl_in if (d[0] > x_cur - halfwidth and d[0] < x_cur + halfwidth)]
 window_y = [d[1] for d in tbl_in if (d[0] > x_cur - halfwidth and d[0] < x_cur + halfwidth)]
 x_vals = [x[0] for x in tbl_in]
 y_vals = [y[1] for y in tbl_in]
 ind_end = x_vals.index(window_x[len(window_x)-1])
# print(len(window_x), len_all, ind_end)
 for j in range(len_all):
  x_cur = tbl_in[j][0] #current x-value 
  x_start = window_x[0]
  x_end = x_cur + halfwidth
  while ind_end < len_all-1: #adding the window end
   x_last = x_vals[ind_end+1]
   y_last = y_vals[ind_end+1]
   window_x.append(x_last)
   window_y.append(y_last) #ok
   if x_last >= x_end: 
    popped_x = window_x.pop(len(window_x)-1)
    popped_y = window_y.pop(len(window_y)-1)
    break
   ind_end += 1
  while x_start < x_cur - halfwidth: #deleting the window beginning
   popped_x = window_x.pop(0)
   popped_y = window_y.pop(0) #   print(popped_x, popped_y)
   x_start = window_x[0] #  print(j, len(window_x), ind_end, '%.3f' % x_start, '%.3f' % x_cur, '%.3f' % x_last, '%.3f' % x_end)
  degree = min(deg, len(window_x) - 1) #smoothing
  if degree == 0:
   val_smoothed = avg(window_y)
  elif degree == 1:
   linfit = list(np.polyfit(window_x, window_y, 1, rcond=None, full=False, w=None, cov=False))
   val_smoothed = linfit[0]*x_cur + linfit[1] #tavgs_fit = [x**2 * sqrfit_tavgs[0] + x*sqrfit_tavgs[1] + sqrfit_tavgs[2] for x in dnums]  
  elif degree == 2:
   sqrfit = list(np.polyfit(window_x, window_y, 2, rcond=None, full=False, w=None, cov=False))
   val_smoothed = sqrfit[0]*x_cur**2 + sqrfit[1]*x_cur + sqrfit[2] #tavgs_fit = [x**2 * sqrfit_tavgs[0] + x*sqrfit_tavgs[1] + sqrfit_tavgs[2] for x in dnums]  
  elif degree == 3:
   cubfit = list(np.polyfit(window_x, window_y, 3, rcond=None, full=False, w=None, cov=False))
   val_smoothed = cubfit[0]*x_cur**3 + cubfit[1]*x_cur**2 + cubfit[2]*x_cur + cubfit[3]
  smoothie.append([x_cur, val_smoothed])
 return smoothie	 

#made separately to achieve continuous smoothing around new year
#@jit(nopython=True)
def smoother_t_stats(tbl_in, hw, deg, *n_iter): 
 tbl_before = [[x[0]-366, x[1]] for x in tbl_in[365-hw:]]
 tbl_after = [[x[0]+366, x[1]] for x in tbl_in[:hw]]
 tbl = tbl_before + tbl_in + tbl_after
 tbl_sm = smoother(tbl, hw, deg)
 tbl_smooth = tbl_sm[hw+1:len(tbl_sm)-hw]
 if n_iter:
  for i in range(n_iter[0] - 1):
   tbl_smooth = smoother_t_stats(tbl_smooth, hw, deg)
 return tbl_smooth

#retrieves all stats from source logs (csv)
#output format for each line: [0] mins, [1] avgs, [2] maxs, [3] precip.
#for each sub-list:
#[0] -  daynum, [1] mth [2] day [3]  avg [4] min [5] year of min [6] max [7] year of max [8] rsd
def daily_vals_all(data_src, yr_st, yr_end, **f_type): 
 if type(data_src) == str: #faster version #OK
  tbl_in = data_prepare(data_src)
 else:
  tbl_in = data_src
 tbl = [x for x in tbl_in if (x[1] < yr_end and x[1] >= yr_st)]
 yr_data = yr_tbl = [[] for i in range(366)]
 for i in range(len(tbl)): #sorting by months and days
  line = tbl[i]
  m = line[2]
  d = line[3]
  d_num = md_to_daynum(m, d) - 1
  yr_data[d_num].append(line)
 for i in range(366): #calculating all daily extremes 
  daydata = yr_data[i]
  m = yr_tbl[i][0][2]
  d = yr_tbl[i][0][3] 
  mins  = [[x[4], x[1]] for x in daydata if x[4] != 'ND'] #x[1] = year
  means = [[x[5], x[1]] for x in daydata if x[5] != 'ND']
  maxs  = [[x[6], x[1]] for x in daydata if x[6] != 'ND']
  precs = [[x[8], x[1]] for x in daydata if x[8] != 'ND']
  mins.sort(key=lambda x:x[0])
  means.sort(key=lambda x:x[0])
  maxs.sort(key=lambda x:x[0])
  precs.sort(key=lambda x:x[0])  
  len_min = len(mins)  #calculate min, avg, max of temps, and respective years. #mins
  min_min = mins[0][0]
  yr_min_min = mins[0][1]
  max_min = mins[len_min-1][0] 
  yr_max_min = mins[len_min-1][1]
  rsd_min, avg_min = rsd([x[0] for x in mins])
  len_mean = len(mins)  #means
  min_mean = means[0][0]
  yr_min_mean = means[0][1]
  max_mean = means[len_mean-1][0] 
  yr_max_mean = mins[len_mean-1][1]
  rsd_mean, avg_mean = rsd([x[0] for x in means])
  len_max = len(mins)  #maxs
  min_max = maxs[0][0]
  yr_min_max = maxs[0][1]
  max_max = maxs[len_min-1][0] 
  yr_max_max = maxs[len_min-1][1]
  rsd_max, avg_max = rsd([x[0] for x in maxs])
  len_prec = len(precs)  #precs
  max_prec = precs[len_prec-1][0] 
  yr_max_prec = precs[len_prec-1][1]
  rsd_prec, avg_prec = rsd([x[0] for x in precs])
  yr_tbl[i] = [[i+1, m, d, avg_min, min_min, yr_min_min, max_min, yr_max_min, rsd_min], 
                [i+1, m, d, avg_mean, min_mean, yr_min_mean, max_mean, yr_max_mean, rsd_mean],  
                [i+1, m, d, avg_max, min_max, yr_min_max, max_max, yr_max_max, rsd_max], 
                [i+1, m, d, avg_prec, 0, 0, max_prec, yr_max_prec, rsd_prec]]
 return yr_tbl, tbl_in 

#separate version for ice, because of different csv format
#output unified with daily_vals_all
def daily_stats_ice(data_src, yr_st, yr_end):
 if type(data_src) == str: #faster version, from daily_vals_all
  tbl_in = []
  os.chdir('source_ice')
  with open(data_src, 'r') as csv:
   lines = csv.readlines()
   for i in range(len(lines)):
    newstr = lines[i].split('\t')
    line = [int(newstr[2]), int(newstr[3]), int(newstr[4]), int(newstr[5]), float(newstr[1]) if newstr[1] != 'ND' else 'ND']
    tbl_in.append(line)
   os.chdir('..')
 else:
  tbl_in = data_src
 for i in range(1, len(tbl_in)-1): #interpolating absent values
  if tbl_in[i][4] == 'ND' and type(tbl_in[i-1][4]) == float and type(tbl_in[i+1][4]) == float:
   tbl_in[i][4] = 0.5 * (tbl_in[i-1][4] + tbl_in[i+1][4]) 
 tbl = [x for x in tbl_in if (x[1] < yr_end and x[1] >= yr_st)]
 yr_data = yr_tbl = [[] for i in range(366)]
 for i in range(len(tbl)): #sorting by months and days
  line = tbl[i]
  m = line[2]
  d = line[3]
  d_num = md_to_daynum(m, d) - 1
  yr_data[d_num].append(line)
 for i in range(366): #calculating all daily extremes 
  daydata = yr_data[i]
  m = yr_tbl[i][0][2]
  d = yr_tbl[i][0][3] 
  exts = [[x[4], x[1]] for x in daydata if x[4] != 'ND']
  exts.sort(key=lambda x:x[0])
  len_ext = len(exts)  #calculate min, avg, max of temps, and respective years. #mins
  min_ext = exts[0][0]
  yr_min_ext = exts[0][1]
  max_ext = exts[len_ext-1][0] 
  yr_max_ext = exts[len_ext-1][1]
  rsd_ext, avg_ext = rsd([x[0] for x in exts])
  yr_tbl[i] = [[i+1, m, d, avg_ext, min_ext, yr_min_ext, max_ext, yr_max_ext, rsd_ext]]
 return yr_tbl, tbl_in

#calculates daily anomaly log of sea ice extent.
def anom_ice(data_src, yr_start, yr_end):
 ice_stats, ice_log = daily_stats_ice(data_src, yr_start, yr_end)
 ice_stats_sm = smoother([[x[0][0], x[0][3]] for x in ice_stats], 2, 0)
 ice_anom_raw = []
 for i in range(len(ice_log)):
  jd = ice_log[i][0]
  yr = ice_log[i][1]
  mth = ice_log[i][2]
  day = ice_log[i][3]
  ext = ice_log[i][4]
  yrday = md_to_daynum(mth, day)
  avg_ext = ice_stats_sm[yrday-1][1]
  if type(ext) == float: #excluding ND
   anom_cur = ext - avg_ext
   ice_anom_raw.append([mjd_to_yr_s(jd), jd, yr, mth, day, anom_cur])
 return ice_anom_raw
 
#calculates daily anomaly logs of temperatures.
#returns also temperature statistics (climatogram)
def anom_temps(data_src, yr_start, yr_end):
 t_stats, t_log = daily_vals_all(data_src, yr_start, yr_end)
 t_stats_sm = smoother_t_stats([[x[1][0], x[1][3]] for x in t_stats], 12, 0, 3)
 t_anom_raw = []
 for i in range(len(t_log)):
  jd = t_log[i][0]
  yr = t_log[i][1]
  mth = t_log[i][2]
  day = t_log[i][3]
  ext = t_log[i][5]
  yrday = md_to_daynum(mth, day)
  avg_ext = t_stats_sm[yrday-1][1]
  if type(ext) == float: #excluding ND
   anom_cur = ext - avg_ext
   t_anom_raw.append([mjd_to_yr_s(jd), jd, yr, mth, day, anom_cur])
 return t_anom_raw, t_log, t_stats, t_stats_sm

#writes smoothed logs  (anomaly, etc)
#takes log_raw because it is created differently for ice and temps.
#half-widths are found in f_type, as well as 'write' option
#f_type: 'halfwidths' - list, 'write' - bool
def smoother_log(log_raw, **f_type):
 if 'halfwidths' in f_type.keys():
  halfwidths = f_type['halfwidths']
 else:
  halfwidths = [1,]
 if 'degrees' in f_type.keys():
  degrees = f_type['degrees']
 else:
  degrees = [0 for x in halfwidths]
 tbl_out = []
 tbl_raw = [[x[0], x[5]] for x in log_raw]
 tbls_sm = []
 for i in range(len(halfwidths)):
  hw = halfwidths[i]
  deg = degrees[i]
  tbls_sm.append(smoother(tbl_raw, hw, deg))
 for i in range(len(log_raw)):
  smoothed = [x[i][1] for x in tbls_sm]
  newstr = log_raw[i] + smoothed
  tbl_out.append(newstr)
 return tbl_out

#creates, writes and returns smoothed anomaly logs (both ice and temps)
#columns: 6 - raw, 7 - month, 8 - half-year, 9 - year, 10 - 2 year smoothed
#full widths are 2 times larger than half-widths! 
def anom_writer(data_src, **f_type): 
 str_name = ''
 prec = ''
 if 'raw' in f_type.keys():
  tbl_in = smoother_log(data_src, **f_type)
 else:
  tbl_in = data_src
 if 'ice' in f_type.keys():
  log_name = 'ice'
  prec = '%.3f'
 elif 'loc' in f_type.keys():
  log_name = str(f_type['loc'])
  prec = '%.2f'
 str_name += log_name + '_anom_log_'
 if 'halfwidths' in f_type.keys():
  halfwidths = f_type['halfwidths']
 else:
  halfwidths = [1,]
 if 'degrees' in f_type.keys():
  degrees = f_type['degrees']
 else:
  degrees = [0 for x in halfwidths]
 if 'nowrite' not in f_type.keys():
  os.chdir('anom_logs')
  str_name += str(halfwidths).replace(', ', '_').strip(']').strip('[') + '.csv'
  with open(str_name, 'w') as csv:
   head_str = log_name + '\n' + 'yr' + '\t\t' + 'MJD' + '\t' + 'yr' + '\t' + 'mth' + '\t' + 'day' + '\t' + 'raw' + '\t' + str(halfwidths).replace(', ', '\t').strip(']').strip('[') + '\n'
   wrt = csv.write(head_str)
   for i in range(len(tbl_in)):
    yr_float = tbl_in[i][0]
    MJD = tbl_in[i][1]
    yr = tbl_in[i][2]
    mth = tbl_in[i][3]
    day = tbl_in[i][4]
    raw = tbl_in[i][5]
    sm = tbl_in[i][6:6+len(halfwidths)]
    newstr =  '%.3f' % yr_float + '\t' + str(MJD) + '\t' + str(yr) + '\t' + str(mth) + '\t' + str(day) + '\t' + prec % raw
    for j in range(len(halfwidths)):
     newstr += '\t' + prec % sm[j]
    newstr += '\n'
    wrt = csv.write(newstr)
  os.chdir('..')
 return tbl_in  
 
 
#makes simple smoothed log in coordinates 'date vs smoothed value'
#reads csv if it is in 'anom logs' folder
#date and values precisions are '%.3f' and '%.1f'. Calculate anew if full precision needed.
def time_log(loc, **f_type):
 sm_type = '2yr' if 'sm_type' not in f_type.keys() else f_type['sm_type'] # raw, mth, half, year, 2yr
 l_cut = 0 if 'l_cut' not in f_type.keys() else f_type['l_cut']
 smtype = smooth_type(sm_type)
 if 'anom_logs' in os.listdir() and 'read' in f_type.keys() and f_type['read'] == True: #read csv
  os.chdir('anom_logs')
  tlog_in = []
  log_name = str(loc) + '_anom_log_0.08_0.5_1_2.csv'
  if log_name in os.listdir():
   tlog_in = []
   with open(log_name, 'r') as csv:
    lines = csv.readlines()
    for i in range(2, len(lines)): #data from line 3
     newstr = lines[i].split('\t')
     date_cur = float(newstr[0]) #years fractional
     val_cur = float(newstr[smtype])  
     tlog_in.append([date_cur, val_cur])   
   os.chdir('..')
  tlog = [x for x in tlog_in if x[0] > l_cut]
 else:  #calculate from source data   
  if loc == 'ice':
   t_an = anom_ice('ice_data.csv', 1981, 1994)
  else:
   t_an, t_log, t_st, t_st_sm = anom_temps(str(loc) + '.csv', 1965, 1985)
  ftype_smth = {'halfwidths': [0.08, 0.5, 1, 2], 'raw': True, 'loc': loc, 'nowrite': True} #write if needed
  t_sm = anom_writer(t_an, **ftype_smth) # 5 raw 6 mth 7 half-year 8 year 9 2yr smoothed
  tlog = [[x[0], x[smtype]] for x in t_sm if x[0] > l_cut] #cut early 1960 anomaly
 return tlog

#----- piecewise fitting custom

#https://stackoverflow.com/questions/29382903/how-to-apply-piecewise-linear-fit-in-python
#k1, k2 - slopes, y0-k1*x0 and y0-k2*x0 - intercepts condition to meet at x0, y0
#to build r2 vs breakpoint list, fix x0 and run it from start to end

#makes 2-piecewise function with break point x0, y0 and slopes k1, k2
def piecewise_linear_2(x, x0, y0, k1, k2):
 return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

#fits 2-piecewise linear function to data
#sometimes unstable if no breakpoint is specified!
#used to built lists of r-squared vs. breakpoint
def piece2fit_scipy(data_in, **f_type): #uses scipy #model: breakpoint left, slope 1, inter 1, bp right
 if 'yr_break' in f_type.keys(): #fixed break point
  yrbrk = f_type['yr_break']
  fit2pc, cov2pc = optimize.curve_fit((lambda x, y0, k1, k2: piecewise_linear_2(x, yrbrk, y0, k1, k2)), [x[0] for x in data_in], [x[1] for x in data_in])
  x_0, y_0, k_1, k_2 = yrbrk, fit2pc[0], fit2pc[1], fit2pc[2]
 else: #straight to optimum #carefully!
  fit2pc, cov2pc = optimize.curve_fit(piecewise_linear_2, [x[0] for x in data_in], [x[1] for x in data_in])
  x_0, y_0, k_1, k_2 = fit2pc[0], fit2pc[1], fit2pc[2], fit2pc[3]
 model = [[0, 
         k_1, 
         y_0 - k_1 * x_0, 
         x_0], #yr_break 
         [x_0, 
         k_2, 
         y_0 - k_2 * x_0, 
         3000]]
 return model


#makes 3-piecewise function with break point x0, y0, x1, y1, and slopes k1, k2 and k3
def piecewise_linear_3(x, x0, y0, x1, y1, k1, k2, k3):
 return np.piecewise(x, [x < x0, (x >= x0) & (x < x1), x >= x1], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0, lambda x:k3*x + y0-k3*x0])

#fits 3-piecewise linear function to data
#use with fixed break points, otherwise unstable
def piece3fit_scipy(data_in, **f_type): #uses scipy #model: breakpoint left, slope 1, inter 1, bp right
 fit3pc, cov3pc = optimize.curve_fit(piecewise_linear_3, [x[0] for x in data_in], [x[1] for x in data_in])
 x_0, y_0, x_1, y_1, k_1, k_2, k_3 = fit3pc[0], fit3pc[1], fit3pc[2], fit3pc[3], fit3pc[4], fit3pc[5], fit3pc[6]
 model = [[0, 
         k_1, 
         y_0 - k_1 * x_0, 
         x_0], #yr_break 
         [x_0, 
         k_2, 
         y_0 - k_2 * x_0, 
         x_1],
         [x_1, 
         k_3, 
         y_0 - k_3 * x_0, 
         3000]]
 return model

#custom
#fits simple linear functions from both sides, 
#calculates predicted values at breakpoints
#inserts line running through both breakpoints
#used to calculate matrices of determination coefficients vs BP1 and BP2
#assumes linear trends before and after transition
#converges to global optimum three-piece-wise fit if BPs are right and middle is not very different from a line
#if yr_break_1 == yr_break_2, converges to two-piece-wise fit with vertical step at breakpoint
#@jit(nopython=True)
def piece3fit(data_in, yr_break_1, yr_break_2): 
 data_1 = [x for x in data_in if x[0] < yr_break_1]
 data_2 = [x for x in data_in if (x[0] > yr_break_1 and x[0] < yr_break_2)]
 data_3 = [x for x in data_in if x[0] > yr_break_2]
 fit_1 = np.polyfit([x[0] for x in data_1], [x[1] for x in data_1], 1, rcond=None, full=True, w=None, cov=False)
 fit_3 = np.polyfit([x[0] for x in data_3], [x[1] for x in data_3], 1, rcond=None, full=True, w=None, cov=False)
 fit_2_l = yr_break_1 * fit_1[0][0] + fit_1[0][1]
 fit_2_r = yr_break_2 * fit_3[0][0] + fit_3[0][1]
 fit_2 = np.polyfit([yr_break_1, yr_break_2], [fit_2_l, fit_2_r], 1, rcond=None, full=True, w=None, cov=False)
 model = [[0, 
           fit_1[0][0], 
           fit_1[0][1], 
           yr_break_1], 
           [yr_break_1, 
           fit_2[0][0], 
           fit_2[0][1], 
           yr_break_2],
           [yr_break_2, 
           fit_3[0][0], 
           fit_3[0][1], 
           3000]]
 return model


#-----piecewise fit calculate r values for all search space and load calculated values
#-----2pc-wise: list of r2 vs break point
#-----3pc-wise: matrix of r2 vs bp1 and bp2


#calculates a list of r2 vs breakpoint in 2-piecewise approx and saves it to dbin file.
#saves r2 vs bp plot if specified in f_type
#Slow! 1 min per 1000 values
def p2fit_calc(loc, n_pts, **f_type):

 place = 'Ice extent' if loc == 'ice' else names[Blashyrkh.index(loc)]
 sm_type = '2yr' if 'sm_type' not in f_type.keys() else f_type['sm_type'] # raw, mth, half, year, 2yr
 
 t_log = time_log(loc, **{'read': True, 'sm_type': sm_type, 'l_cut': 1963}) 
 
 r_list = []
 date_beg = t_log[0][0] 
 date_end = t_log[len(t_log)-1][0]
 
 for i in range(1, n_pts):
  date_cur = date_beg + (date_end - date_beg) * i / n_pts
  model_cur = piece2fit_scipy(t_log, **{'yr_break': date_cur}) #1 per 3 sec, too bad
  rsqr = r_sqr(t_log, model_cur)
  r_list.append([date_cur, model_cur, rsqr[1]])
 
 r_list_srt = sorted(r_list, key=lambda x: -x[2])
 yr_max = r_list_srt[0][0]
 r2_max = r_list_srt[0][2]
 
 if 'savebin' in f_type.keys() and f_type['savebin'] == True:
  os.chdir('p2fit')
  bin_name = 'fit2_' + sm_type + '_' + str(loc) + '.dbin' 
  with open(bin_name, 'wb') as bfile:
   pickle.dump(r_list, bfile)
  os.chdir('..')
  	 
 if 'draw' in f_type.keys() and f_type['draw'] == True:
  plt.clf() #  plt.ion() #  plt.draw()
  plt.plot([x[0] for x in r_list], [y[2] for y in r_list])
  plt.grid(True)
  title_str_1 = 'Ice extent anomaly 2pc-wise fit, R2 vs break point\n' if loc == 'ice' else str(loc) + ' ' + str(place) + ' temperature anomaly 2pc-wise fit, R2 vs break point\n'
  title_str_2 = 'Smooth type: ' + sm_type + '; R2 max: ' + '%.4f' % r2_max + ' at year ' + '%.2f' % yr_max
  plt.title(title_str_1 + title_str_2)

  if 'save' in f_type.keys() and f_type['save'] == True:
   os.chdir('p2fit')
   img_title = 'fit2_' + sm_type + '_' + str(loc)
   plt.savefig(img_title)
   os.chdir('..')
 return r_list
 

#calculates an r2 vs bp1 and bp2 matrix (list of values) in 3-piecewise linear fit
#vesion with string sm_type and other args - in polarfit_all.
#Slow! 20 calcs per second, need usually 10-50k points
def r2_calc(loc, n_pts, bp1_st, bp1_end, bp2_st, bp2_end, **f_type): # 5 raw 6 mth 7 half-year 8 year 9 2yr smoothed

 sm_type = '2yr' if 'sm_type' not in f_type.keys() else f_type['sm_type']
 prefix = '' if 'sm_type' not in f_type.keys() else (f_type['prefix'])
 smtype = smooth_type(sm_type)
 save_bin = False if 'nosave' in f_type.keys() and f_type['nosave'] == True else True
 t_log = time_log(loc, **{'read': False, 'sm_type': sm_type, 'l_cut': 1963})

 r_matrix = []
 ctr = 0
 ctr_print_step = min(int(n_pts/4), 10000)
 t_0 = time.time()
 while ctr < n_pts:# and loc in (20046, 20069, 20087, 20292, 20674, 20891): #25 per second
  yr1 = random.uniform(bp1_st, bp1_end) #initially 1965 - 2022
  yr2 = random.uniform(bp2_st, bp2_end)
  yr_1 = min(yr1, yr2)
  yr_2 = max(yr1, yr2)
  model_cur = piece3fit(t_log, yr_1, yr_2)
  r_corr = r_sqr(t_log, model_cur)[1]
  if r_corr < 1 and r_corr > 0:
   ctr += 1
   r_matrix.append([yr_1, yr_2, model_cur, r_corr])
   if ctr % ctr_print_step == 0:
    print(loc, ctr, yr_1, yr_2, r_corr, time.time() - t_0)
 
 if save_bin:
  os.chdir('r2 ' + sm_type)
  bin_file = prefix + ('_' if len(prefix) != 0 else '') + str(loc) + '_r2_sm_' + sm_type + '_' + str(int(n_pts/1000)) + 'k.dbin'
  with open (bin_file, 'wb') as rm:
   pickle.dump(r_matrix, rm)
  os.chdir('..')
 
 return r_matrix

#loads r-matrix for 3pcwise approx from .dbin file
#adds r_matrices of the same type and locations together, to save calculation time
#returns (as calculated, sorted by r2 value)
def r2_load(loc, sm_type, **f_type):
 r_matrix_in = []
 os.chdir('r2 '+ sm_type)
 all_files = os.listdir()
 for fn in all_files: #collect all r_matrices for loc and smooth_type
  if 'dbin' in fn and str(loc) in fn and sm_type in fn: #for both ice and temps
   with open (fn, 'rb') as rm:
    r_matr = pickle.load(rm) #0 - break 1, [1] - break 2, [2] - model, [3] - color temp
    r_matrix_in += r_matr 
 os.chdir('..')
 if 'trunc' in f_type.keys(): #specify trunc condition like 'trunc': (lambda x, y: x > 1990 and x < 2010 and y > 1995 and y < 2015)
  tf = f_type['trunc']
  r_matrix_1 = [x for x in r_matrix_in if tf(x[0], x[1])]
 else:
  r_matrix_1 = [x for x in r_matrix_in if x[0] > 1975] if loc in (25051, 25282) else r_matrix_in   
 r_matrix = [x for x in r_matrix_1 if x[3] < 1 and x[3] > 0] # and x[0] > 1995 and x[0] < 2010 and x[1] < 2017 and x[1] > x[0] + 3] #r_matrix = [x for x in r_matrix_1 if not (x[3] > 1 or x[3] < 0 or (x[0] < 1985 and x[1] > 2015) or (x[0] < 1970 and loc == 21432)  or (x[1] > 2018 and loc == 21982))] 
  
 len_r = len(r_matrix)
 r_mat = sorted(r_matrix, key=lambda f: -f[3]) 
 return r_matrix, r_mat

#loads r2 vs breakpoint lists for two-piece-wise approximation
def r2list_load(loc, sm_type, **f_type):
 os.chdir('p2fit')
 fn = 'fit2_' + sm_type + '_' + str(loc) + '.dbin'
 with open (fn, 'rb') as rl:
  r_list_in = pickle.load(rl) #0 - break 1, [1] - break 2, [2] - model, [3] - color temp
 if 'trunc' in f_type.keys(): #specify trunc condition like 'trunc': (lambda x, y: x > 1990 and x < 2010 and y > 1995 and y < 2015)
  tf = f_type['trunc']
  r_list_1 = [x for x in r_list_in if tf(x[0])]
 r_list = [x for x in r_list_1 if x[2] < 1 and x[2] > 0] # and x[0] > 1995 and x[0] < 2010 and x[1] < 2017 and x[1] > x[0] + 3] #r_matrix = [x for x in r_matrix_1 if not (x[3] > 1 or x[3] < 0 or (x[0] < 1985 and x[1] > 2015) or (x[0] < 1970 and loc == 21432)  or (x[1] > 2018 and loc == 21982))] 
 r_list_srt = sorted(r_list, key=lambda f: -f[2])
 os.chdir('..')
 return r_list, r_list_srt
 

#===== data plot

#plots raw and smoothed anomalies and saves them to png according to f_type
#used for saving to png and interactive plotting
#accepts offsets on Y-axis
def anom_plot(loc, **f_type): #add yr_start, yr_end in f_type
 place_name = names[Blashyrkh.index(loc)]
 if loc == 'ice':
  t_an = anom_ice('ice_data.csv', 1981, 1994)
 else:
  t_an, t_log, t_st, t_st_sm = anom_temps(str(loc) + '.csv', 1965, 1985)
 no_wrt = False if 'stats_write' in f_type.keys() and f_type['stats_write'] == True else True
 ftype_smth = {'halfwidths': [0.08, 0.5, 1, 2], 'raw': True, 'loc': loc, 'nowrite': no_wrt}
 t_sm = anom_writer(t_an, **ftype_smth) # 5 raw 6 mth 7 half-year 8 year 9 2yr smoothed
 sm_type = 'mth half year 2yr' if 'sm_type' not in f_type.keys() else f_type['sm_type']
 yr_color = '#ff8020' if 'yr_color' not in f_type.keys() else f_type['yr_color']
 lst = 'solid' if 'linestyle' not in f_type.keys() else f_type['linestyle']
 off_set = f_type['offset'] if 'offset' in f_type.keys() else 0  

 if 'raw' in sm_type:
  plt.plot([x[0] for x in t_sm], [off_set + x[5] for x in t_sm], linewidth=1, linestyle=lst, color = '#d0f0f0', )
 if 'mth' in sm_type:
  plt.plot([x[0] for x in t_sm], [off_set + x[6] for x in t_sm], linewidth=1, linestyle=lst, color='#a0a0ff')
 if 'half' in sm_type: 
  plt.plot([x[0] for x in t_sm], [off_set + x[7] for x in t_sm], linewidth=1, linestyle=lst, color='green')
 if 'year' in sm_type: 
  plt.plot([x[0] for x in t_sm], [off_set + x[8] for x in t_sm], linewidth=1, linestyle=lst, color=yr_color)
 if '2yr' in sm_type: 
  plt.plot([x[0] for x in t_sm], [off_set + x[9] for x in t_sm], linewidth=2, linestyle=lst, color='red')# plt.xlim(x_limit)# plt.ylim(-2.5, 6.5)
 title_str = 'Ice extent anomalies' if loc == 'ice' else str(loc) + ' ' + place_name + ' temperature anomalies'
 title_str_2 = 'smooth window half-width: none (cyan), month(blue), half-year(green), year(yellow), two years (red)'
 plt.title(title_str)
 plt.suptitle(title_str_2, **{'x': 0.5, 'y': 0.05}) 
 if 'x_lims' in f_type.keys():
  plt.xlim(f_type['x_lims'])
 if 'y_lims' in f_type.keys():
  plt.ylim(f_type['y_lims'])
 plt.grid(True)
 
 if 'save' in f_type.keys() and f_type['save'] == True:
  dir_name = 'pics_t_anoms'
  if dir_name not in os.listdir():
   os.mkdir(dir_name)
  os.chdir(dir_name)
  img_suffix = f_type['img_suffix'] if 'img_suffix' in f_type.keys() else ''
  img_prefix = f_type['img_prefix'] if 'img_prefix' in f_type.keys() else ''
  img_name = img_prefix + '_' + str(loc) + ('_ext_' if loc == 'ice' else '_temp_') + 'anoms_' + img_suffix + '.jpg'  
  plt.gcf().set_size_inches(11, 6)
  plt.savefig(img_name, dpi=200)
  os.chdir('..')
 
 if 'plot' in f_type.keys() and f_type['plot'] == True: 
  plt.ion()
  plt.show()
 else:
  plt.close()
 
 return None


#used to calculates two linear fits to anomaly logs - before and after the tipping point
def fit_compare_linear(loc, sm_type, yr1st, yr1end, yr2st, yr2end):
 t_log = time_log(loc, **{'sm_type': sm_type, 'read': True})
 lin_before = np.polyfit([x[0] for x in t_log if (x[0] > yr1st and x[0] < yr1end)], [x[1] for x in t_log if (x[0] > yr1st and x[0] < yr1end)], 1, rcond=None, full=True, w=None, cov=False)
 lin_after  = np.polyfit([x[0] for x in t_log if (x[0] > yr2st and x[0] < yr2end)], [x[1] for x in t_log if (x[0] > yr2st and x[0] < yr2end)], 1, rcond=None, full=True, w=None, cov=False)
 model_before = [[yr1st, lin_before[0][0], lin_before[0][1], yr1end],]
 model_after  = [[yr2st, lin_after[0][0], lin_after[0][1], yr2end],]
 yr_mid = 2004
 val_mid_1 = yr_mid * lin_before[0][0] + lin_before[0][1]
 val_mid_2 = yr_mid * lin_after[0][0] + lin_after[0][1]
 jump_val = val_mid_2 - val_mid_1
 return model_before, model_after, jump_val, val_mid_1, val_mid_2

#slow if no 'read'! takes 20 sec
#used to calculate uncertainties of linear fits to anomaly logs
def fit_compare_linear_stats(loc, sm_type):
 model_list = [] 
 for i in range(100):
  yr1_st = random.uniform(1960, 1965)
  yr1_end = random.uniform(1995, 2000)
  yr2_st = random.uniform(2007, 2011)
  yr2_end = random.uniform(2019, 2023)
  fits = fit_compare_linear(loc, 'mth', yr1_st, yr1_end, yr2_st, yr2_end)
  model_list.append([fits[0][0][0], fits[0][0][1], fits[0][0][2], fits[0][0][3], fits[1][0][0], fits[1][0][1], fits[1][0][2], fits[1][0][3], fits[2], fits[3], fits[4]])
 st_k1 = rsd([x[1] for x in model_list])
 st_b1 = rsd([x[2] for x in model_list])
 st_k2 = rsd([x[5] for x in model_list])
 st_b2 = rsd([x[6] for x in model_list])
 st_j  = rsd([x[8] for x in model_list])
 st_v1 = rsd([x[9] for x in model_list])
 st_v2 = rsd([x[10] for x in model_list])
 return st_k1, st_b1, st_k2, st_b2, st_j, loc, st_v1, st_v2

#plots piecewise and linear fits, writes r2, intercepts and slopes, saves a png
#takes best piecewise fits brom r2 matrices and lists loaded from .dbin files
#to plot non-global-best 3pcwise fit, truncate r2-matrix manually around local optimum
#specify truncating conditions by lambda-function in **f_type of r2_load
#calculates linear fit
def fit_compare_simple(loc, sm_type, **f_type):
 filenames = os.listdir()
 place = 'Ice extent' if loc == 'ice' else names[Blashyrkh.index(loc)]
 yr_start = 1965
 yr_end = 1985
 ftype = {'halfwidths': [0.08, 0.5, 1, 2], 'raw': True, 'loc': loc, 'nowrite': True}

 smtype = smooth_type(sm_type)

 t_log = time_log(loc, **{'read': True, 'sm_type': sm_type, 'l_cut': 1965})

#---r2_matrix of 3pc-wise fit
 if 'trunc_3' in f_type.keys(): 
  r_matrix, r_mat = r2_load(loc, sm_type, **{'trunc': f_type['trunc_3']}) #loads truncated if needed (lambda condition)
 else:
  r_matrix, r_mat = r2_load(loc, sm_type) #loads truncated if needed (lambda condition)
 	 
 len_r_3 = len(r_matrix)
 r_mat = sorted(r_matrix, key=lambda f: -f[3])
 r_max_3 = r_mat[0][3]
 best_fit_3 = r_mat[0][2]
 bp1_bf3 = r_mat[0][0] #break points of best fit
 bp2_bf3 = r_mat[0][1]
 if 'r2 plot' in f_type.keys() and f_type['r2 plot'] == True:
  f_type_r2plot = {'colorcode': 'blues', 'base': 0, 'pow': 6, 'rel': True, 'rel_scale': 1.0, 'sm_type': sm_type, 'trunc': trunc_3}
  rmtr, rmt = r2_plot(loc, **f_type_r2plot, **{'save': True})

#---r2_list of 2pc-wise fit
 if 'trunc_2' in f_type.keys(): #typically cut between 1975 and 2019
  r_list, r_lst_srt = r2list_load(loc, sm_type, **{'trunc': f_type['trunc_2']}) #loads truncated if needed (lambda condition)
 else:
  r_list, r_lst_srt = r2list_load(loc, sm_type) #loads truncated if needed (lambda condition)
 	 
 len_r_2 = len(r_list)
 r_max_2 = r_lst_srt[0][2]
 bp_bf2 = r_lst_srt[0][0]
 best_fit_2 = r_lst_srt[0][1] 

#---linear fit
 lin_fit = np.polyfit([x[0] for x in t_log], [x[1] for x in t_log], 1, rcond=None, full=True, w=None, cov=False)
 model_lin = [[0, lin_fit[0][0], lin_fit[0][1], 3000],]

 r_corr_1, data_calc_1 = tuple(r_sqr(t_log, model_lin)[1:])
 r_corr_2, data_calc_2 = tuple(r_sqr(t_log, best_fit_2)[1:])
 r_corr_3, data_calc_3 = tuple(r_sqr(t_log, best_fit_3)[1:])

#---from fit_plot, adapted
 slope_3_1, inter_3_1 = best_fit_3[0][1], best_fit_3[0][2] #move in graph_plot
 slope_3_2, inter_3_2 = best_fit_3[1][1], best_fit_3[1][2]
 slope_3_3, inter_3_3 = best_fit_3[2][1], best_fit_3[2][2]

 slope_2_1, inter_2_1 = best_fit_2[0][1], best_fit_2[0][2] #move in graph_plot
 slope_2_2, inter_2_2 = best_fit_2[1][1], best_fit_2[1][2]

 slope_lin, inter_lin = model_lin[0][1], model_lin[0][2]

 model_lin_str = 'linear:  ' + ' R2 adj = ' + '%.2f' % (r_corr_1*1000) + '\n ' + '%.0f' % inter_lin + ' + ' + '%.3f' % slope_lin + ' * yr'
 
 model_2_str = '2-spline:  ' + ' R2 adj = ' + '%.2f' % (r_corr_2*1000) + '\n break pt ' + '%.2f' % bp_bf2
  
 model_3_r_str = '3-spline: R2 adj = ' + '%.2f' % (r_corr_3*1000) + '\n\n breakpoints: '  + '%.2f' % bp1_bf3 + ', ' + '%.2f' % bp2_bf3
 model_3_1_str = '  spline 1: ' + '%.2f' % inter_3_1 + ' + ' + '%.3f' % slope_3_1 + ' * yr'
 model_3_2_str = '  spline 2: ' + '%.2f' % inter_3_2 + ' + ' + '%.3f' % slope_3_2 + ' * yr'
 model_3_3_str = '  spline 3: ' + '%.2f' % inter_3_3 + ' + ' + '%.3f' % slope_3_3 + ' * yr'
 
 begin_str = str(loc) + ' ' + str(place) + ' temperature anomaly fit results (base 1965 - 1985)' if loc != 'ice' else ' Ice extent anomaly fit results (base 1981 - 1994)'
 title_str = 'smooth window halfwidth: ' + sm_type
 title_string = begin_str + '\n' + title_str
 stat_str = 'slope 1 ' + '%.3f' % slope_3_1 + 'slope 1 ' + '%.3f ' % slope_3_2 + ' slope 3 ' + '%.3f' % slope_3_3
 img_name = str(loc) + 'anomaly fits_'# + 'r2_' + '%.2f' % (r_corr*1000) 
 plt.plot([x[0] for x in t_log], [x[1] for x in t_log], linewidth=2, color='#8060ff')
 plt.plot([x[0] for x in data_calc_1], [x[1] for x in data_calc_1], linewidth=1, linestyle='dashed', color='#00ff00')
 plt.plot([x[0] for x in data_calc_2], [x[1] for x in data_calc_2], linewidth=1, linestyle='dashed', color='#00ff00')
 plt.plot([x[0] for x in data_calc_3], [x[1] for x in data_calc_3], linewidth=2, linestyle='dashed', color='#ff8060')
 plt.title(title_string)
 (x_l, y_l) = (0.72, 0.14) if loc != 'ice' else (0.15, 0.14)
 plt.figtext(x_l, y_l, model_lin_str + '\n\n' + model_2_str + '\n\n' + model_3_r_str + '\n' + model_3_1_str + '\n' + model_3_2_str + '\n' + model_3_3_str)
 plt.axis('tight')
 plt.xlim(1960, 2025)
 y_min = min([x[1] for x in t_log if x[0] < 2025 and x[0] > 1960])
 y_max = max([x[1] for x in t_log if x[0] < 2025 and x[0] > 1960])
 plt.ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))
 plt.ylabel('ice extent anomaly' if loc == 'ice' else 'temperature anomaly')
 plt.grid(True)
 
 if 'save' in f_type.keys() and f_type['save'] == True:
  suffix = '_' + f_type['suffix'] if 'suffix' in f_type.keys() else ''
  dir_name = 'pics_fitcompare_' + sm_type
  if dir_name not in os.listdir():
   os.mkdir(dir_name)
  os.chdir(dir_name)
  img_name = 'fits_' + ('_' if loc == 'ice' else '') + str(loc) + '_' + ('_' if sm_type == 'year' else '') + sm_type + suffix
  plt.gcf().set_size_inches(13, 6)
  plt.savefig(img_name, dpi=200)  
  os.chdir('..')
  dir_name = 'pics_fitcompare_all'
  if dir_name not in os.listdir():
   os.mkdir(dir_name)
  os.chdir(dir_name)
  plt.savefig(img_name, dpi=200)  
  os.chdir('..')

 
 if 'plot' in f_type.keys() and f_type['plot'] == True: 
  plt.ion()
  plt.show()
 else:
  plt.close()
 return {'r_corr_1': r_corr_1, 'model_1': model_lin, 'data_calc_1': data_calc_1, 'r_corr_2': r_corr_2, 'model_2': best_fit_2, 'data_calc_2': data_calc_2, 'r_corr_3': r_corr_3, 'model_3': best_fit_3, 'data_calc_3': data_calc_3, 'data_in': t_log}

#makes a color-coded diagram of r2 matrix for 3pc-wise fit
#loads matrix with r2_load function
def r2_plot(loc, sm_type, **f_type): #f_type: colorcode, sm_type, pow, nodraw, save, rel, base, heat_max, pt_size
 
 place = 'Ice extent' if loc == 'ice' else names[Blashyrkh.index(loc)]
 colorcode = 'bw' if 'colorcode' not in f_type.keys() else f_type['colorcode'] #bw_rel, 'bw', 'heat'
 powr = 1 if 'pow' not in f_type.keys() else f_type['pow']
 rel = False if 'rel' not in f_type.keys() else f_type['rel']
 rel_scale = 1 if 'rel_scale' not in f_type.keys() else f_type['rel_scale'] #1.1: 1.1 * r_max
 r_base = 0 if 'base' not in f_type.keys() else f_type['base']
 heat_max = 21500 if 'heat_max' not in f_type.keys() else f_type['heat_max']  
 r_matrix, r_mat = r2_load(loc, sm_type, **f_type) #can load truncated by lambda condition, see r2_load
 if 'trunc' not in f_type.keys() or ('plot_lim' in f_type.keys() and f_type['plot_lim'] == 'default'):
  plot_lims = (1960, 2025, 1960, 2025) 
 else:
  plot_lims = (min([x[0] for x in r_matrix]), max([x[0] for x in r_matrix]), min([x[1] for x in r_matrix]), max([x[1] for x in r_matrix])) 
 len_r = len(r_matrix)
 
 r_max = r_mat[0][3]
 invert = False if 'invert' not in f_type.keys() else True
 
 r_scale = r_max * rel_scale if rel else 1
 pt_size = int(sqrt(2e5/len_r) * 6) + 1 if 'pt_size' not in f_type.keys() else f_type['pt_size']  

 if colorcode == 'heat':
  for i in range(len(r_mat)):
   r_f = r_mat[i][3]
   color_temp = heat_max - ((r_f - r_base) / (r_scale - r_base)) ** powr * (heat_max - 1500)  # (-10 if r_f < 0 else (log(r_f) + 5.1) * 4e4)
   pt_color = '#' + t_to_rgb(color_temp)
   if len(r_mat[i]) >= 5:
    r_mat[i][4] = pt_color
   else:
    r_mat[i].append(pt_color)
 elif colorcode == 'blues': # bw
  for i in range(len(r_mat)):
   r_f = r_mat[i][3]
   gscl = ((r_f - r_base) / (r_scale - r_base)) ** powr * 255  # (-10 if r_f < 0 else (log(r_f) + 5.1) * 4e4)
   r = g = 64 + int(gscl*0.75)
   b = 255
   r_hex = hex(int(r))[2:].zfill(2)
   g_hex = hex(int(g))[2:].zfill(2)
   b_hex = hex(int(b))[2:].zfill(2)
   pt_color = '#' + r_hex + g_hex + b_hex
   if len(r_mat[i]) >= 5:
    r_mat[i][4] = pt_color
   else:
    r_mat[i].append(pt_color)
 else: # bw
  for i in range(len(r_mat)):
   r_f = r_mat[i][3]
   gscl = ((r_f - r_base) / (r_scale - r_base)) ** powr * 255  # (-10 if r_f < 0 else (log(r_f) + 5.1) * 4e4)
   r = g = 64 + int(gscl)*0.75
   b = 255
   pt_color = '#' + hex(int(gscl))[2:].zfill(2) * 3
   if len(r_mat[i]) >= 5:
    r_mat[i][4] = pt_color
   else:
    r_mat[i].append(pt_color)
   
 best_fit = r_mat[0]
 yr1 = best_fit[0]
 yr2 = best_fit[1]
 r_fac = best_fit[3]
 slope_1 = best_fit[2][0][1] #move in graph_plot
 slope_2 = best_fit[2][1][1]
 slope_3 = best_fit[2][2][1]
 
 plt.clf()
 r_mat[0][4] = '#ffb0b0' #mark the best point
 
 r_mat.sort(key=lambda x: x[3]) #draw from worst to best
 plt.scatter([x[0] for x in r_mat], #break point 1
             [y[1] for y in r_mat], #break point 2
             c=[c[4] for c in r_mat], #color
             marker='o',
             alpha=1, #not transparent
             s=pt_size)
 plt.xlim(plot_lims[0], plot_lims[1]) 
 plt.ylim(plot_lims[2], plot_lims[3])
 title_str_1 = 'Ice extent anomaly R2 vs break points diagram\n' if loc == 'ice' else str(loc) + ' ' + str(place) + ' temperature anomaly R2 vs break points diagram\n'
 title_str_2 = 'Smooth type: ' + sm_type
 plt.title(title_str_1 + title_str_2)
 plt.xlabel("Break point 1")
 plt.ylabel("Break point 2")
 plt.rc('grid', linestyle= ':', color='gray')
 plt.grid(True)
 plt.suptitle('Best fit with breakpoints: ' + '%.2f' % yr1 + ', '  + '%.2f' % yr2 + ',\n R squared: ' + '%.3f' % r_fac, **{'x': 0.6, 'y': 0.25})
 if 'save' in f_type.keys() and f_type['save'] == True:
  img_name = 'r2_' + ('__' if loc == 'ice' else '_') + str(loc) + ('__' if sm_type == 'year' else '_') + sm_type + '_' + str(int(len_r/1000)) + 'k'
  plt.gcf().set_size_inches(11, 6)
  os.chdir('r2 ' + sm_type)
  plt.savefig(img_name, dpi=200)
  os.chdir('..')
 if 'show' in f_type.keys() and f_type['show'] == True:
  plt.ion()
  plt.draw()
 r_mat.sort(key=lambda x: -x[3]) #return sorted from best to worst
 return r_matrix, r_mat

#makes r-matrix and fitcompare plots
#Use to auto-generate unified plots
def all_plot(loc, **f_type):
 r_matrix, r_mat = r2_plot(loc, **f_type)
 yr_1, yr_2 = r_mat[0][0], r_mat[0][1]
 out_data = fit_compare_simple(loc, **f_type)
 return None 


#plots climatogram for chosen location, reference period, and smoothing type specified in **f_type.
#output format for t_stats: [0] mins, [1] avgs, [2] maxs, [3] precip.
#for each sub-list:
#[0] -  daynum, [1] mth [2] day [3]  avg [4] min [5] year of min [6] max [7] year of max [8] rsd
#t_stats_sm returns only one column

def climatogram(loc, yr_start, yr_end, **f_type):
 loc_name = names[Blashyrkh.index(loc)]
 loc_src = str(loc) + '.csv'
 halfwidth = 12 if 'hw' not in f_type.keys() else f_type['hw']
 poly = 0 if 'poly' not in f_type.keys() else f_type['poly']
 n_iter = 3 if 'n_iter' not in f_type.keys() else f_type['n_iter']
 t_stats, t_log = daily_vals_all(loc_src, yr_start, yr_end) 
 min_sm = smoother_t_stats([[x[0][0], x[0][3]] for x in t_stats], halfwidth, poly, n_iter)
 avg_sm = smoother_t_stats([[x[1][0], x[1][3]] for x in t_stats], halfwidth, poly, n_iter)
 max_sm = smoother_t_stats([[x[2][0], x[2][3]] for x in t_stats], halfwidth, poly, n_iter)
 maxmax = smoother_t_stats([[x[2][0], x[2][6]] for x in t_stats], halfwidth, poly, n_iter)
 minmin = smoother_t_stats([[x[0][0], x[0][4]] for x in t_stats], halfwidth, poly, n_iter)
 avg_pr = smoother_t_stats([[x[3][0], x[3][3]] for x in t_stats], halfwidth, poly, n_iter)
 av_rsd = smoother_t_stats([[x[1][0], x[1][8]] for x in t_stats], halfwidth, poly, n_iter)
 pr_rsd = smoother_t_stats([[x[3][0], x[3][8]] for x in t_stats], halfwidth, poly, n_iter)
 ylims1 = (min([x[0][4] for x in t_stats]) - 5, max([x[2][6] for x in t_stats]) + 5) if 't_scale' not in f_type.keys() else f_type['t_scale']
 ylims2 = (0, 1.2 * max(max([x[1] for x in avg_pr]), max([x[1] for x in av_rsd]) ) ) if 'rsd_scale' not in f_type.keys() else f_type['rsd_scale']
 plt.clf()
 fig, ax1 = plt.subplots()
 ax1.plot([x[0] for x in min_sm], [x[1] for x in min_sm], linewidth=2, color='#8080ff')
 ax1.plot([x[0] for x in avg_sm], [x[1] for x in avg_sm], linewidth=2, color='#20a020')
 ax1.plot([x[0] for x in max_sm], [x[1] for x in max_sm], linewidth=2, color='#ff8040')
 ax1.plot([x[0] for x in maxmax], [x[1] for x in maxmax], linewidth=2, linestyle='dotted', color='#e00000')
 ax1.plot([x[0] for x in minmin], [x[1] for x in minmin], linewidth=2, linestyle='dotted', color='#0000e0')
 ax1.plot([x[0][0] for x in t_stats], [x[0][3] for x in t_stats], color='#8080ff', alpha=0.2, marker='^', linewidth=1, markersize=2)
 ax1.plot([x[1][0] for x in t_stats], [x[1][3] for x in t_stats], color='#40c040', alpha=0.2, marker='^', linewidth=1, markersize=2)
 ax1.plot([x[2][0] for x in t_stats], [x[2][3] for x in t_stats], color='#ff8040', alpha=0.2, marker='^', linewidth=1, markersize=2)
 ax1.plot([x[2][0] for x in t_stats], [x[2][6] for x in t_stats], color='#d04020', alpha=0.2, marker='^', linewidth=1, markersize=2)
 ax1.plot([x[0][0] for x in t_stats], [x[0][4] for x in t_stats], color='#4020d0', alpha=0.2, marker='^', linewidth=1, markersize=2)
 ax2 = ax1.twinx()
 ax2.plot([x[0]+1 for x in avg_pr], [x[1] for x in avg_pr], linewidth=3, color='#b0d0ff')
 ax2.plot([x[0]+1 for x in av_rsd], [x[1] for x in av_rsd], linewidth=3, linestyle='dashed', color='#80b080')
 #ax2.plot([x[0] for x in pr_rsd], [x[1] for x in pr_rsd], linewidth=1, color='#ffffff')
 title_string_1 = str(loc) + ' ' + loc_name + ' climatogram base ' + str(yr_start) + ' - ' + str(yr_end) + '\n'
 title_string_2 = "halfwidth: " + str(halfwidth) + ' poly: ' + str(poly) + ' n_iter: ' + str(n_iter)
 title_string = title_string_1 + title_string_2
 plt.title(title_string)
 plt.xlim(1, 366)
 ax1.set_ylim(ylims1)
 ax2.set_ylim(ylims2)
 ax1.grid(visible=True, which='both', axis='y')
 d0 = 1
 for i in range(len(m_lengths)):
  plt.axvline(d0, color='gray')
  ax2.text(d0 + 10, 0.02*ylims2[1], m_names[i], fontsize=12)
  d0 += m_lengths[i]
 if 'save' in f_type.keys() and f_type['save'] == True:
  res_png = f_type['dpi'] if 'dpi' in f_type.keys() else 200
  img_name = str(loc) + '_' + loc_name + '_' + str(yr_start) + '_' + str(yr_end) + '_' + 'climatogram'
  plt.gcf().set_size_inches(11, 6)
  plt.savefig(img_name, dpi=res_png)
  plt.close()
 else:
  plt.ion()
  plt.show()
 return None


#manual search of three-piece-wise fit R2 maximum
#keyboard control:
# bp1: a/d move left/right, q/e incr/decr shift step
# bp2: [/] move left/right, -/= incr/decr shift step
def manual_fit(loc, **f_type): #OK
 sm_type = f_type['sm_type'] if 'sm_type' in f_type.keys() else '2yr'
 savefig = True if 'savefig' in f_type.keys() and f_sype['savefig'] == True else False
 place = names[Blashyrkh.index(loc)]
 l_cut = f_type['l_cut'] if 'l_cut' in f_type.keys() else 0 
 t_log = time_log(loc, **{'sm_type': sm_type, 'l_cut': l_cut})
 plt.ion()
 plt.draw()
 shift_factor_1 = shift_factor_2 = 1
 cont = 'y'
 yr_1 = yr1_max = 1990
 yr_2 = yr2_max = 2010
 r2_max = 0
 while cont != 'n':
  cont = input('lets play around\n')
  if cont == 'a':
   yr_1 -= shift_factor_1
  elif cont == 'd':
   yr_1 += shift_factor_1
  elif cont == '[':
   yr_2 -= shift_factor_2
  elif cont == ']':
   yr_2 += shift_factor_2
  elif cont == 'q':
   shift_factor_1 /= 2
  elif cont == 'e':
   shift_factor_1 *= 2
  elif cont == '-':
   shift_factor_2 /= 2
  elif cont == '=':
   shift_factor_2 *= 2
  elif cont == 's': #clear values
   r2_max, yr1_max, yr2_max = r_corr, yr_1, yr_2
   shift_factor_2 *= 2
  plt.clf()
  model_cur = piece3fit(t_log, yr_1, yr_2)
  r_corr, data_calc = tuple(r_sqr(t_log, model_cur)[1:]) 
  if r_corr > r2_max:
   r2_max, yr1_max, yr2_max, model_best = r_corr, yr_1, yr_2, model_cur
  img_name = str(loc) + '_base_65_85_' + 'r2_' + '%.2f' % (r_corr*1000) 
  plt.plot([x[0] for x in t_log], [x[1] for x in t_log], linewidth=2, color='green')
  plt.plot([x[0] for x in data_calc], [x[1] for x in data_calc], linewidth=1, color='blue')
  title_str_1 = str(loc) + ' ' + place + ' ' + ' t anom 3pc-wise fit\n'
  title_str_2 = 'R2 ' + '%.2f' % (r_corr*1000) + '   break 1 ' + '%.2f' % yr_1 + '   break 2 ' + '%.2f' % yr_2 + '\n'
  title_str_3 = 'R2max ' + '%.2f' % (r2_max*1000) + '   break 1 ' + '%.2f' % yr1_max + '   break 2 ' + '%.2f' % yr2_max 
  plt.title( title_str_1 + title_str_2 + title_str_3)
  (x_l, y_l) = (0.72, 0.2) if loc != 'ice' else (0.15, 0.2)

  slope_3_1, inter_3_1 = model_cur[0][1], model_cur[0][2] #move in graph_plot
  slope_3_2, inter_3_2 = model_cur[1][1], model_cur[1][2]
  slope_3_3, inter_3_3 = model_cur[2][1], model_cur[2][2]
  
  model_3_r_str = 'current fit:\n'
  model_3_1_str = '  spline 1: ' + '%.2f' % inter_3_1 + ' + ' + '%.3f' % slope_3_1 + ' * yr'
  model_3_2_str = '  spline 2: ' + '%.2f' % inter_3_2 + ' + ' + '%.3f' % slope_3_2 + ' * yr'
  model_3_3_str = '  spline 3: ' + '%.2f' % inter_3_3 + ' + ' + '%.3f' % slope_3_3 + ' * yr'

  plt.figtext(x_l, y_l, model_3_r_str + '\n' + model_3_1_str + '\n' + model_3_2_str + '\n' + model_3_3_str)
  plt.axis('tight')
  plt.grid(True)
 if savefig:
  plt.savefig(img_name)
 return model_best


#=====supplementary

#returns column index with needed smooth type in log prepared by anom_temps
def smooth_type(sm_type): 
 if sm_type == 'raw':
  smtype = 5
 elif sm_type == 'mth':
  smtype = 6
 elif sm_type == 'half':
  smtype = 7
 elif sm_type == 'year':
  smtype = 8
 else: #2yr and default
  smtype = 9
 return smtype
