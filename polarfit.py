#ver.2 rewritten for numpy

#functions
#for actual scripts, see polarfit_scripts
#Analysis of daily temperature logs from weather stations and sea ice extent logs
#data extraction, analysis and plotting

#all source data are located in src_dir
#place ice/other data into src_dir manually if needed
#output data folders are generated automatically


#adding math functions
#all other needed modules are imported with math_custom

#frequently used function arguments: 

#base=(1965, 1980)- time span on which daily averages are calculated. These daily averages are used in anomaly calculation: anomaly = value - daily average (i.e. temperature @ December, 21th )
#log_range=(1965,2020)- time span included in calculations (full log is truncated to this range)
#sm_anom=(183,1,1)- smoothing parameters: smoothing window half-width, degree of polynomial fitted to data in smoothing window, number of passes
#sm_stats=(12,1,3) - smoothing parameters of daily averages (i.e. climatograms)

import sys
dir_code = 'D:\\Code\\math_custom'
if dir_code not in sys.path:
 sys.path.insert(0, dir_code) 

from math_custom import *


import requests
from bs4 import BeautifulSoup

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

#list of locations to include in calculation
#txt file: WMO numbers of polar meteo stations, location names
with open('locs.txt', 'r') as f:
 strs = f.readlines()
locs = [x.strip('\n').split(', ') for x in strs][:]
Blashyrkh, names = [x[0] for x in locs], [x[1] for x in locs][:]

m_lengths = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
m_len_sums = np.hstack((0, np.cumsum(m_lengths)[:-1]))
m_names = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
sm_types = ['raw', 'mth', '2m', 'half', 'year', '2yr', '4yr'] #sm_types = ['2yr', 'year', 'half', 'mth', 'raw'] #half-widths of smoothing windows used in the following data processing
smooth_hws = np.array([nan, 15, 30, 91, 183, 366, 732])

#indices of values in source data and daily stats (i.e. in daily stats table, [365,i_st_rsd] = root square deviation of value at 366th day, Dec, 31)
len_stats=9
i_st_nd, i_st_m, i_st_d, i_st_avg, i_st_rsd, i_st_min, i_st_yr_min, i_st_max, i_st_yr_max = np.arange(len_stats)
i_src_ice, i_src_t_min, i_src_t_avg, i_src_t_max, i_src_prec = 4, 4, 5, 6, 8
i_st_t_min, i_st_t_avg, i_st_t_max, i_st_prec = 0, 1, 2, 3
arr_dummy = np.empty(0)
src_dir = 'source_data'

t_noon = np.array('12:00:00')
jd_unix = 2440587.5
dt_unix = np.datetime64('1970-01-01 00:00:00')
d_current = dt.now()
yr_cur, mth_cur = d_current.timetuple()[0:2]
sm_stats = (12,1,3) 
sm_anom = (183,1,1) 



#=========================================================
#==================== math ===============================
#=========================================================


#calculates determination coefficient (R-squared simple and adjusted) for data and a model
# takes piece-wise model in form [[x_start_1, slope_1, intercept_1, x_end_1], [x_start_2, ... ]]
#as returned by fitting functions (see below)
#also returns calculated values according to model
#@jit(nopython=True)
def r2_calc(ts, vals, model):
 n_pts = vals.size
 n_df = 1 + model.shape[0]
 simple_avg = np.nanmean(vals)
 vals_calc = piecewise_eval(ts, model)
 res = vals - vals_calc
 res_total = vals - simple_avg
 sqrs_fit = np.nansum(res**2)
 sqrs_total = np.nansum(res_total**2)
 det_coeff = 1 - sqrs_fit / sqrs_total
 det_coeff_adj = 1 - (1 - det_coeff**2) * (n_pts - 1) / (n_pts - n_df)
 return det_coeff, det_coeff_adj, np.array((ts, vals_calc)).T

 
#=========================================================
#===============data extraction/processing================
#========================================================= 

 
#basic_site = 'http://pogodaiklimat.ru/monitor.php?id=' + '27612' + '&month=2&year=2022' 
#weather_page = requests.get(basic_site, verify=False).text
#extracts table of temperatures from a single month and a single site
def weather_tbl(wpage): #(+)
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
   row_cur.append(float(columns[i].text) if val != '' else nan)
   row_cur[0] = int(row_cur[0])
  weather_table.append(row_cur)
 return np.array(weather_table) #day, tmin, tavg, tmax, tanom, precip

  
#creates continuous weather record from a single site, using by weather_tbl function
#returns year, month, day, t_min, t_avg, t_max, t_anom, precipitation
#writes data to .csv at src_dir
#t_anoms are not used and recalculated later, before any other data processing
def weather_extract(loc, yr_start=0, yr_end=yr_cur, ftype=''): #(+)
 mth_end = 12 if yr_end < yr_cur else mth_cur
 if not loc.isdigit(): #ice area or other data
  return None
 place = names[Blashyrkh.index(loc)]
 csv_name = loc + '.csv'
 fn = src_dir + '/' + csv_name
 if src_dir not in os.listdir():
  os.mkdir(src_dir)
 tbl_prev_present = csv_name in os.listdir(src_dir)
 if tbl_prev_present and 'reload' not in ftype:
  tbl_all = [np.genfromtxt(fn, comments='#', delimiter='\t', skip_header=0, missing_values=nan, filling_values=nan, usecols=np.arange(8)),]
  yr_last, mth_last = tbl_all[0][-1,0], tbl_all[0][-1,1] + 1
 else:
  tbl_all=[]
  yr_last, mth_last = yr_start, 1
 if yr_last > yr_cur or (yr_last == yr_cur and mth_last >= mth_cur):
  return None
 wpage_0 = f"http://pogodaiklimat.ru/monitor.php?id={loc}&month=1&year=2023"
 weather_page = requests.get(wpage_0, verify=False).text
 soup = BeautifulSoup(weather_page, 'html.parser')
 t_span = soup.find('ul', {'class': 'climate-list'})
 paragr = t_span.find_all('li')
 allspans = paragr[len(paragr)-1].find_all('span') 
 yr = max(yr_start, yr_last, int(allspans[len(allspans)-1].text[:4]))
 mth = max(1, mth_last) #get starting year (beginning of records)
 print(f"Extracting weather data for {place} ({loc})")	
 t_0 = time.time()
 cont=True
 while cont:
  time.sleep(random.uniform(0.5, 2.5))
  if mth == 13:
   yr, mth = yr + 1, 1.0
  weather_site = f"http://pogodaiklimat.ru/monitor.php?id={int(loc)}&month={int(mth)}&year={int(yr)}"
  print(f"{time.time() - t_0:.2f} {int(yr)} {int(mth)}")
  try:
   w_cur_0 = weather_tbl(weather_site)
  except ConnectionError:
   break
  n_rows = w_cur_0.shape[0]
  w_cur = np.hstack((np.ones((n_rows,1))*yr, np.ones((n_rows,1))*mth, w_cur_0 ))
  mask_finite = np.isfinite(w_cur_0[:,1:]).sum(axis=1).astype(bool)
  if not np.any(mask_finite) or yr > yr_end or (yr == yr_end and mth >= mth_end): #add data until current month
   cont=False
  else:
   tbl_all.append(w_cur[mask_finite])
   mth += 1
  
 tbl_out = np.vstack(tbl_all)
 np.savetxt(fn, tbl_out, fmt='%.1f', delimiter='\t', newline='\n')
 return tbl_out


  
#------------------- data preparation ---------------------

#reads csv data from drive (all types: weather/ice area/volume
#deletes nan-containing/non-unique lines, adds julian day numeration #
def data_load(src, log_range =(-inf, inf), ftype=''): #OK  (r+) #50kpoints/s
 fn = loc_to_filename(src) 
 if 'Ice extent' in fn:
  vs_all = np.genfromtxt(fn, delimiter=',')
  vs_0 = vs_all[1:,2:-3].T #crop m, d, separator 1st, separator last, mean, median 
  areas_0 = vs_0.flatten()
  ds_list = vs_all[1:,1]
  ms_list = np.cumsum(np.hstack((1.0, np.where(np.diff(ds_list) < 0, 1, 0))))
  n_yrs = vs_0.shape[0]
  yrs_list = 1978 + np.arange(n_yrs)
  yrs_0 = np.repeat(yrs_list, 366)
  ms_0 = np.tile(ms_list, n_yrs) 
  ds_0 = np.tile(ds_list, n_yrs) 
  nan_mask = np.where(np.isfinite(areas_0))
  yrs, ms, ds, areas = yrs_0[nan_mask], ms_0[nan_mask], ds_0[nan_mask], areas_0[nan_mask]
  dates = ymd_to_date(yrs, ms, ds)
  jds = (dates - dt_unix) / np.timedelta64(1, 'D') + jd_unix#  jds = ymd_to_jd(np.array(yrs, ms, ds)) # np.zeros_like(ds)  for i in range(ds.size):   jds[i] = ymd_to_jd((int(yrs[i]), int(ms[i]), int(ds[i]))) + 0.5
  vs = np.vstack((jds, yrs, ms, ds, areas)).T #OK #interpolate nans if needed
 elif 'Ice volume' in fn:
  vs_all = np.genfromtxt(fn, skip_header=1)
  n_pts = vs_all.shape[0]
  yrs, daynums, vols = vs_all.T
  jd_0 = ymd_to_jd(np.vstack((yrs, np.ones(n_pts), np.ones(n_pts))).T)
  jds = jd_0 + daynums - 1
  leap_mask = np.where((yrs % 4 == 0) * (daynums >= 60) )
  jds[leap_mask] = jds[leap_mask] + 1
  ymds = jd_to_ymd(jds)
  vs_0 = np.hstack((jds[:,None], ymds, vols[:,None]*1000))
 
  inds_3_01 = np.where((vs_0[:,2]==3) * (vs_0[:,3]==1) * (vs_0[:,1] % 4 == 0))[0]
  inds_2_28 = inds_3_01 - 1
  vols_2_29 = 0.5 * (vs_0[inds_3_01,-1] + vs_0[inds_2_28,-1])
  yrs_2_29 = vs_0[inds_3_01, 1] 
  ymds_add = np.vstack((yrs_2_29, np.full_like(yrs_2_29, 2), np.full_like(yrs_2_29, 29) )).T
  jds_add = ymd_to_jd(ymds_add)
  vs_add = np.hstack((jds_add[:,None], ymds_add, vols_2_29[:,None]))
  vs_added = np.vstack((vs_0, vs_add))
  vs = vs_added[np.argsort(vs_added[:,0])]
  jds = vs[:, 0]
 else: 
  vs_0 = np.genfromtxt(fn, delimiter='\t')[:,:]
  nan_mask = np.where(np.isnan(vs_0[:,3:].sum(axis=1)), False, True)
  vs = vs_0[nan_mask]# yrs, mths, ds = vs[:,0], vs[:,1], vs[:,2]
  jds = ymd_to_jd(vs[:,:3]) # np.zeros(vs.shape[0])  for i in range(jds.size):   jds[i] = date_to_jd(tuple(vs[i,:3].astype(int))) + 0.5 #afternoon of each day
  vs = np.hstack((jds[:,None], vs))
 yrs_x = jd_to_yr(jds) #add np.unique on yr_mask here
 yr_mask = inrange_arr(yrs_x, log_range)
 if 'yrs' in ftype:
  vs[:,0] = yrs_x
 vs_masked = vs[yr_mask]
 
 return vs_masked[np.unique(vs_masked[:,0], return_index=True)[1]]

#extracts raw time series of value indicated by i_data from csv file
def t_log(src, i_data=i_src_t_avg, ftype='', log_range=None):
 vs = data_load(src, log_range=log_range) if isinstance(src, int) or isinstance(src, str) else src
 ts = vs[:,0] if 'yrs' not in ftype else jd_to_yr(vs[:,0])
 vals = vs[:, i_data]
 return ts, vals 


#prepares anomaly logs used in fit calculation and visualization
#loc = number of weather station (i.e. 20069), or 'ice' if ice area is calculated
def t_log_calc(src, log_range=(1963,2025), base=(1960, 1985), sm_stats=sm_stats, sm_anom=(183,1,1), i_data=-1, ftype=''):

 #choose anomaly smoothing type: standard if indicated in ftype; else - according to sm_anom variable.
 hw, smt = smooth_hw(ftype)
 if not isnan(hw):
  hw_an, smt = (hw, smt)
  sm_anom = (hw_an, 1, 1)
 elif sm_anom==None:
  hw_an, smt = 1, 'raw'
 else:
  hw_an, smt = (sm_anom[0], 'year')
  
 loc_input =  not isinstance(src, np.ndarray)
 weather_calc = loc_input and (isinstance(src, int) or src.isdigit())
 if weather_calc:
  base = (1965,1980) if base==None else base
  i_data = i_src_t_avg if i_data < 0 else i_data
 else:
  base = (1981,1994) if base==None else base
  i_data = i_src_ice if i_data < 0 else i_data
 
  #load and prepare data
 tbl = data_load(src) if loc_input else src
 ts_0, vs_raw_0 = t_log(tbl, i_data=i_data, ftype='')
 yrs_0 = jd_to_yr(ts_0)

 if 'raw only' in ftype:
  log_mask = np.isfinite(vs_raw_0) * inrange_arr(yrs_0, log_range)
  return (yrs_0[log_mask] if 'jds' not in ftype else ts_0[log_mask]), vs_raw_0[log_mask]

 vals_0, daily_avgs = anom_calc(ts_0, vs_raw_0, base=base, sm_stats=sm_stats, sm_an=sm_anom, ftype=ftype + ' full')
 mask_fin = np.isfinite(vals_0)
 yrs_IR = inrange_arr(yrs_0, log_range) * inrange_arr(yrs_0, (yrs_0[0], yrs_0[-1])) # + 0.5*hw_an/366 #truncate also edge effects after smoothing
 log_mask = mask_fin * yrs_IR 

 ts, vs_raw, yrs, vals = ts_0[log_mask], vs_raw_0[log_mask], yrs_0[log_mask], vals_0[log_mask]
 ts_out = ts if 'jd' in ftype else yrs
 
 if 'full' in ftype:
  return ts_out, vals, vs_raw, daily_avgs
 elif 'raw' in ftype:
  return ts_out, vals, vs_raw
 else:
  return ts_out, vals


#--------------------data_processing-------------------

#add ice area data read and prepare here

#calculates stats for single day values for different years:
#[0] -  daynum, [1] mth [2] day [3]  avg [4] min [5] year of min [6] max [7] year of max [8] rsd
#i_st_nd, i_st_m, i_st_d, i_st_avg, i_ts_rsd, i_st_sm, i_st_min, i_st_yr_min, i_st_max, i_st_yr_max = np.arange(10)
def daily_stats_single(yrs, vs):
 i_val_min, i_val_max = np.nanargmin(vs), np.nanargmax(vs)
 val_min, val_max = vs[i_val_min], vs[i_val_max]
 yr_min, yr_max = yrs[i_val_min], yrs[i_val_max]
 val_rsd, val_avg = np.nanstd(vs), np.nanmean(vs)
 return val_avg, val_rsd, val_min, yr_min, val_max, yr_max


#calculates all daily stats for weather or ice area data 
#output format for each line: [0] t_min, [1] t_avg, [2] t_max, [3] precip. (weather), [0]: ice area
#in ver.1, nan values were interpolated (not needed now?)
def daily_stats(ts, vals, base=(1960,1985), ftype=''): #(r+) #400kpts/s (250 with jd_to_ymd_conversion)
 yrs, ms, ds = jd_to_ymd(ts).T if ts.ndim==1 else ts.T
 yr_tbl = np.zeros((366, 9))
 mask_yr = np.where((yrs < base[1]) * (yrs >= base[0]), True, False)
 for n_d in range(366): 
  m, d = daynum_to_md(n_d+1)
  mask_md = np.where( (ms == m) * (ds == d), True, False)
  mask_calc = mask_yr * mask_md 
  yrs_md = yrs[mask_calc]
  vals_md = vals[mask_calc]
  yr_tbl[n_d] = np.hstack((n_d+1, m, d, daily_stats_single(yrs_md, vals_md)))
 return yr_tbl

#smoothes vals_avg for day 1-366; uses around-new-year expansion to eliminate edge effects
def daily_stats_smooth(vals_avg, sm=sm_stats):
 hw, deg, n_iter = sm
 nds_exp, vals_avg_exp = np.arange(366*3) + 1, np.hstack((vals_avg, vals_avg, vals_avg))
 vals_smooothed = smoother(vals_avg_exp, xs=nds_exp, hw=hw, deg=deg, n_iter=n_iter, ftype=('convolve' if deg==0 else ''))
 return vals_smooothed[366:732]

#calculates all temperature stats for given location
def stats_temp(tbl, base=(1960,1985), sm=sm_stats):
 ts, Ts_min, Ts_avg, Ts_max, precs = tbl.T[[0, i_src_t_min, i_src_t_avg, i_src_t_max, i_src_prec]]
 t_min_stats = daily_stats(ts, Ts_min, base=base)
 t_avg_stats = daily_stats(ts, Ts_avg, base=base)
 t_max_stats = daily_stats(ts, Ts_max, base=base)
 prec_stats  = daily_stats(ts, precs, base=base)
 st = np.array((t_min_stats, t_avg_stats, t_max_stats, prec_stats)) #np.concatenate((t_min_stats[:,None,:], t_avg_stats[:,None,:], t_max_stats[:,None,:], prec_stats[:,None,:]), axis=1)
 return st

#calculates anomalies (absolute values - daily averages) from time log julian days or ymds)
def anom_calc(ts, vals, dailystats=arr_dummy, base=(1965, 1985), sm_stats=sm_stats, sm_an=(183, 1, 1), ftype=''): #300kpts/s with stats
 yrs, ms, ds = jd_to_ymd(ts).T if ts.ndim==1 else ts.T
 yrs = jd_to_yr(ts) 
 st = daily_stats(ts, vals, base=base) if dailystats.shape[0]!=366 else dailystats
 nds_st, val_avg = st[:,0].astype(int), st[:, i_st_avg]
 val_sm = daily_stats_smooth(val_avg, sm=sm_stats)
 mds = daynum_to_md(nds_st)
 anoms = np.zeros_like(vals)
 for i_d in range(st.shape[0]):
  nd, val_ref = nds_st[i_d], val_sm[i_d]
  m, d = daynum_to_md(nd)
  i_md = np.where((ds==d) * (ms==m))[0]
  anoms[i_md] = vals[i_md] - val_ref
 if sm_an != None and sm_an[0] > 1:
  hw_an, deg_an, n_iter_an = sm_an
  anoms = smoother(anoms, xs=yrs, hw=hw_an/366, deg=deg_an, n_iter=n_iter_an, ftype=ftype)
 if 'plt' in ftype:
  plt.ion()
  plt.show()
  if 'clf' in ftype:
   plt.close('all')
  plt.plot(yrs, anoms)
  plt.grid(True)
 if 'full' in ftype:
  return anoms, val_sm
 else:
  return anoms



#=========================================================
#===============data modelling ===========================
#=========================================================



#https://stackoverflow.com/questions/29382903/how-to-apply-piecewise-linear-fit-in-python
#k1, k2 - slopes, y0-k1*x0 and y0-k2*x0 - intercepts condition to meet at x0, y0
#to build r2 vs breakpoint list, fix x0 and run it from start to end



#------------------piecewise fit custom ------------------------

#piecewise models: numpy arrays in the following form:
#[[left boundary, slope, intercept, right boundary],...]
#shape (n, 4) where n = numper of linear intervals
#nth left boundary == n-1th right boundary

#-----create piecewise functions-----

#makes 2-piecewise function with break point x0, y0 and slopes k1, k2
def piecewise_linear_2(x, x0, y0, k1, k2):
 return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

#makes 3-piecewise function with break point x0, y0, x1, y1, and slopes k1, k2 and k3
def piecewise_linear_3(x, x0, y0, x1, y1, k1, k2, k3):
 return np.piecewise(x, [x < x0, (x >= x0) & (x < x1), x >= x1], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0, lambda x:k3*x + y0-k3*x0])

#-------piecewise fitting-------

#---scipy

#fits 2-piecewise linear function to data
#sometimes unstable if no breakpoint is specified!
#used to built lists of r-squared vs. breakpoint
def piece2fit_scipy(ts, vals, yr_break=nan): #uses scipy #model: breakpoint left, slope 1, inter 1, bp right
 if isfinite(yr_break): #fixed break point
  fit2pc, cov2pc = optimize.curve_fit((lambda x, y0, k1, k2: piecewise_linear_2(x, yr_break, y0, k1, k2)), ts, vals)
  x_0, y_0, k_1, k_2 = yr_break, fit2pc[0], fit2pc[1], fit2pc[2]
 else: #straight to optimum #carefully!
  fit2pc, cov2pc = optimize.curve_fit(piecewise_linear_2, ts, vals)
  x_0, y_0, k_1, k_2 = fit2pc[0], fit2pc[1], fit2pc[2], fit2pc[3]
 model = np.array([[-inf, k_1, y_0 - k_1 * x_0, x_0], #yr_break 
         [x_0, k_2, y_0 - k_2 * x_0, inf]])
 return model


#fits 3-piecewise linear function to data
#use with fixed break points, otherwise unstable
def piece3fit_scipy(ts_0, vals_0, t_break_1, t_break_2): #uses scipy #model: breakpoint left, slope 1, inter 1, bp right
 mask_fin = np.isfinite(ts_0) * np.isfinite(vals_0)
 ts, vals = ts_0[mask_fin], vals_0[mask_fin]
 fit3pc, cov3pc = optimize.curve_fit(piecewise_linear_3, ts, vals)
 x_0, y_0, x_1, y_1, k_1, k_2, k_3 = fit3pc[0], fit3pc[1], fit3pc[2], fit3pc[3], fit3pc[4], fit3pc[5], fit3pc[6]
 model = np.array([[-inf,  k_1, y_0 - k_1 * x_0, x_0], #yr_break 
         [x_0, k_2, y_0 - k_2 * x_0, x_1],
         [x_1, k_3, y_0 - k_3 * x_0, inf]])
 return model

#---custom

#fits simple linear functions from both sides, 
#calculates predicted values at breakpoints
#inserts line running through both breakpoints
#used to calculate matrices of determination coefficients vs BP1 and BP2
#assumes linear trends before and after transition
#if yr_break_1 == yr_break_2, converges to two-piece-wise fit with vertical step at breakpoint
#@jit(nopython=True)
def piece3fit(ts_0, vals_0, t_break_1, t_break_2): 
 mask_fin = np.isfinite(ts_0) * np.isfinite(vals_0)
 ts, vals = ts_0[mask_fin], vals_0[mask_fin]
 mask_1 = np.where(ts < t_break_1, True, False)
 mask_3 = np.where(ts > t_break_2, True, False) # mask_2 = True ^ (mask_1 + mask_3)
 fit_1 = np.polyfit(ts[mask_1], vals[mask_1], 1, rcond=None, full=True, w=None, cov=False)
 fit_3 = np.polyfit(ts[mask_3], vals[mask_3], 1, rcond=None, full=True, w=None, cov=False)
 fit_2_l = t_break_1 * fit_1[0][0] + fit_1[0][1]
 fit_2_r = t_break_2 * fit_3[0][0] + fit_3[0][1]
 fit_2 = np.polyfit([t_break_1, t_break_2], [fit_2_l, fit_2_r], 1, rcond=None, full=True, w=None, cov=False)
 model = np.array([[-inf, fit_1[0][0], fit_1[0][1], t_break_1], 
           [t_break_1, fit_2[0][0], fit_2[0][1], t_break_2],
           [t_break_2, fit_3[0][0], fit_3[0][1], inf]])
 return model

#--------piecewise model evaluating and plotting

def piecewise_eval(x, md):
 n_pc = md.shape[0]
 ks, bs = md[:, 1], md[:,2]
 bps = np.hstack((md[0,0], md[:, -1] ))
 conds = [inrange_arr(x, (bps[i], bps[i+1])) for i in range(n_pc)]
 fs = [(lambda y, i=i: ks[i] * y + bs[i]) for i in range(n_pc)]
 return np.piecewise(x, conds, fs)

def piecewise_plot(md, t_lims=(1965,2025), lstyle=(2, 'dashed', '#8060ff'), ftype=''): #check
 xs = np.hstack((t_lims[0], md[:-1,-1], t_lims[1]))
 ys = piecewise_eval(xs, md)
 if 'clf' in ftype:
  plt.close('all')
 if lstyle==None:
  plt.plot(xs, ys)
 else:
  plt.plot(xs, ys, linewidth=lstyle[0], linestyle=lstyle[1], color=lstyle[2])
 return None

#-------data fitting------------

#calculation of linear piecewise fits to anomaly data 
#to find possible abrupt changes two- or three-piecewise fits are compared by adjusted R2 criterion
#abrupt change is found if a fit with two close break points is better than all fits with single break point
#use nan-filtered time series with range containing BP search range


#calculates best three piece-wise linear fit, varying position of two break points separated by bp_min_diff or more
#generates a set of breakpoints; chooses best 10% of solutions, creates new BP set around selented BPs, 
#iterates until convergence (BP spread no more than thr) or ineration number limit (ctr_lim)
#n_pts = number of randomly generated BP sets at each iteration
#converges poorly if there's no distinct three ranges
#slow, 10-100 s per call
def piece3fit_best_calc(yrs, vals, bp_range=(1965,2020), thr=0.3, bp_min_diff=1.0, n_pts=300, ctr_lim=10, ftype='', cond=None): # 5 raw 6 mth 7 half-year 8 year 9 2yr smoothed  
 yr_min, yr_max = yrs.min(), yrs.max()
 bp_lim_l = max(bp_range[0], yr_min + 2)
 bp_lim_r = min(bp_range[1], yr_max - 2)

 bps = np.linspace(bp_lim_l, bp_lim_r, int(n_pts**0.6)) #initial array of break points
 bps_1, bps_2 = np.meshgrid(bps, bps)
 bps_arr_0 = np.vstack(( bps_1.flatten(), bps_2.flatten())).T
 mask_diff = np.where(bps_arr_0[:,1] - bps_arr_0[:,0] > bp_min_diff, True, False)
 mask_cond = np.ones(bps_arr_0.shape[0]) if cond == None else np.where(cond(bps_arr_0[:,0], bps_arr_0[:,1]), True, False)
 bps_arr = bps_arr_0[np.nonzero(mask_diff * mask_cond)]
 yr_spread = bp_lim_r - bp_lim_l
 ctr=-1
 while yr_spread > thr and ctr < ctr_lim:
  ctr += 1
  r2s = r2_list(yrs, vals, bps_arr) 
  i_sort_r2 = np.argsort(-r2s)
  bp_1, bp_2 = bps_arr[i_sort_r2[0]]
  r2_best = r2s[i_sort_r2[0]]
  bps_best = bps_arr[i_sort_r2[:np.sqrt(r2s.size).astype(int)]]
  yr_spread = max(bps_best[:,0].std(), bps_best[:,1].std())
  n_mult = int(n_pts / bps_best.shape[0])
  bps_0 = np.tile(bps_best, (n_mult,1))
  rand_add = np.random.normal(0, yr_spread, (bps_0.shape[0],2))
  bps_next = bps_0 + rand_add
  mask_diff = np.where((bps_next[:,1] - bps_next[:,0] > bp_min_diff), True, False) * inrange_arr(bps_next[:,0], bp_range) * inrange_arr(bps_next[:,1], bp_range)
  mask_cond = np.ones(bps_next.shape[0]) if cond == None else np.where(cond(bps_next[:,0], bps_next[:,1]), True, False)
  bps_arr = bps_next[np.nonzero(mask_diff * mask_cond)] #  bps_arr = bps_next[np.where(bps_next[:,1] - bps_next[:,0] > bp_min_diff, True, False) * inrange_arr(bps_next[:,0], bp_range) * inrange_arr(bps_next[:,1], bp_range) ]
  
 md = piece3fit(yrs, vals, bp_1, bp_2)
 
 return bp_1, bp_2, r2_best, md

#selects best solution from a matrix of R2 coefficients, calculated in "break pt 1 - break pt 2" coordinates (see below)
#fn - stored R2 matrix filename
#ftype: 'recalc' - calculate new R2 matrix
def piece3fit_best_mat(yrs, vals, bp_range=(1965,2020), fn='', cond=None, ftype=''):
 r2_mat, bps = r2_load(fn=fn, dir_r2='r2s')
 if 'calc' in ftype or r2_mat.size == 0:
  r2_mat, bps = r2_calc_all_3(yrs, vals, bp_range=bp_range, bp_min_diff=0.0, res_yr=0.1 if '_30' in fn else 0.2, ftype='')
 bps_1, bps_2 = np.meshgrid(bps, bps, indexing='ij')
 mask_r2 = np.zeros_like(r2_mat, dtype=np.bool_) if cond == None else True ^ cond(bps_1, bps_2)
 r2_m = np.copy(r2_mat)
 r2_m[mask_r2] = nan
 i_1, i_2 = np.unravel_index(np.nanargmax(r2_m), r2_m.shape)
 r2_best = r2_mat[i_1, i_2]
 bp_1, bp_2 = bps[i_1], bps[i_2]
 md = piece3fit(yrs, vals, bp_1, bp_2)
 return bp_1, bp_2, r2_best, md


#calculates best two piece-wise linear fit, varying position of a single break point
def piece2fit_best_calc(yrs, vals, bp_range=(1965,2020), thr=0.3, n_pts=100, ctr_lim=10): # 5 raw 6 mth 7 half-year 8 year 9 2yr smoothed
 yr_min, yr_max = yrs.min(), yrs.max()
 bps = np.linspace(bp_range[0], bp_range[1], n_pts) #initial array of break points
 r2s = r2_list(yrs, vals, bps)
 yr_spread = bps.max() - bps.min()
 ctr=-1
 while yr_spread > thr and ctr < ctr_lim:
  ctr += 1
  r2s = r2_list(yrs, vals, bps) 
  i_sort_r2 = np.argsort(-r2s)
  bps_best = bps[i_sort_r2[:np.sqrt(r2s.size).astype(int)]]
  bp = bps[i_sort_r2[0]]
  r2_best = r2s[i_sort_r2[0]]
  yr_spread = bps_best.std()
  n_mult = int(n_pts / bps_best.shape[0])
  bps_0 = np.tile(bps_best, n_mult)
  rand_add = np.random.normal(0, yr_spread, bps_0.shape[0])
  bps_next = bps_0 + rand_add
  bps = bps_next[inrange_arr(bps_next, bp_range)]
 
 md = piece2fit_scipy(yrs, vals, bp)
 
 return bp, r2_best, md

  
#manual search for optimal three-piece-wise fit, by R2 criterion
#takes time log of temperature/ice area anomaly

# bp1: a/d move left/right, q/e incr/decr shift step
# bp2: [/] move left/right, -/= incr/decr shift step
def manual_fit(ts, vals, ftype=''): #OK
 hw, smt = smooth_hw(ftype)
 savefig = 'save' in ftype 
 plt.ion()
 plt.draw()
 shift_factor_1 = shift_factor_2 = 1
 cont = 'y'
 yr_1 = yr1_max = 1990
 yr_2 = yr2_max = 2010
 r2_max = 0
 str_cur = 'let\'s play around...\n a/d , [/]: shift left,right breakpoint to the left/right \n q/e , -/= : increase/decrease step for left, right breakpoint \n s: clear values, n: finish '
 while cont != 'n':
  cont = input(str_cur)
  if cont == 'a':
   yr_1 -= shift_factor_1
   str_cur = f"break point 1: {yr_1:.2f}\n"
  elif cont == 'd':
   yr_1 += shift_factor_1
   str_cur = f"break point 1: {yr_1:.2f}\n"
  elif cont == '[':
   yr_2 -= shift_factor_2
   str_cur = f"break point 2: {yr_2:.2f}\n"
  elif cont == ']':
   yr_2 += shift_factor_2
   str_cur = f"break point 2: {yr_2:.2f}\n"
  elif cont == 'q':
   shift_factor_1 /= 2
   str_cur = f"break pt 1 step: {shift_factor_1:.2e} yr\n"
  elif cont == 'e':
   shift_factor_1 *= 2
   str_cur = f"break pt 1 step: {shift_factor_1:.2e} yr\n"
  elif cont == '-':
   shift_factor_2 /= 2
   str_cur = f"break pt 2 step: {shift_factor_2:.2e} yr\n"
  elif cont == '=':
   shift_factor_2 *= 2
   str_cur = f"break pt 2 step: {shift_factor_2:.2e} yr\n"
  elif cont == 's': #clear values
   shift_factor_1, shift_factor_2 = 1, 1, 1990, 2010
   str_cur = f"bps: {bp_1:.2e}, {bp_2:.2e}; steps 1 yr\n"
  plt.clf()
  md = piece3fit(ts, vals, yr_1, yr_2)
  r2, r2_adj, vals_calc = r2_calc(ts, vals, md)
  if r2_adj > r2_max:
   r2_max, yr1_max, yr2_max, model_best = r2_adj, yr_1, yr_2, md
  loc = 'Blashyrkh'
  img_name = f"{loc}_manual_fit" 

  plt.plot(ts, vals, linewidth=2, color='green')
  plt.plot(vals_calc[:,0], vals_calc[:,1], linewidth=1, color='blue')

  title_str_1 = f"{loc} t anom 3pc-wise fit\n" #" str(loc) + ' ' + place + ' ' + '
  title_str_2 = f"R2 {(r2_adj*1000):.2f}, break 1: {yr_1:.2f}, break 2: {yr_2:.2f}" 
  title_str_3 = f"R2 max {(r2_max*1000):.2f}, break 1: {yr1_max:.2f}, break 2: {yr2_max:.2f}" 
  plt.title( title_str_1 + '\n' + title_str_2 + '\n' + title_str_3)
  (x_l, y_l) = (0.72, 0.2) if 'Ice' not in loc else (0.15, 0.2)

  slope_3_1, slope_3_2, slope_3_3 = md[:,1]
  inter_3_1, inter_3_2, inter_3_3 = md[:,2]

  model_str = f"current fit:\n  spline 1: {inter_3_1:.2f} + {slope_3_1:.3f} * yr\n  spline 2: {inter_3_2:.2f} + {slope_3_2:.3f} * yr\n spline 3: {inter_3_3:.2f} + {slope_3_3:.3f} * yr" 
 
  plt.figtext(x_l, y_l, model_str)
  plt.axis('tight')
  plt.grid(True)
 if savefig:
  plt.savefig(img_name)
 return model_best


#-----piecewise fit: calculate r values for all search space or load calculated values
#-----2pc-wise: list of r2 vs break point
#-----3pc-wise: matrix of r2 vs bp1 and bp2


#calculates a list of r2 vs breakpoint for 2-piecewise fit or (yrs, vals) time series.
def r2_calc_all_2(yrs, vals, bp_range=(1965,2020), res_yr=0.3, ftype=''):
 bps = np.arange(bp_range[0], bp_range[1], res_yr)
 n_pts = bps.size
 r2_arr = np.zeros(n_pts)*nan
 for i_bp in range(n_pts):
  yr_break = bps[i_bp]
  md = piece2fit_scipy(yrs, vals, yr_break=yr_break)
  r2_cur = r2_calc(yrs, vals, md)[1]
  r2_arr[i_bp] = r2_cur
 return r2_arr, bps
  

#calculates an r2 vs break pt 1 and break pt 2 matrix for 3-piecewise linear fit
#Slow! 400s per location (500*500 pts) 
def r2_calc_all_3(yrs=arr_dummy, vals=arr_dummy, bp_range=(1965,2020), bp_min_diff=1.0, res_yr=0.1, ftype='', dir_r2='r2s', vals_name=''): # 5 raw 6 mth 7 half-year 8 year 9 2yr smoothed 
 bps = np.arange(bp_range[0], bp_range[1], res_yr)
 bps_1, bps_2 = np.meshgrid(bps, bps, indexing='ij')
 calc_mask = np.where(bps_2 - bps_1 > bp_min_diff, True, False)
 n_pts_1, n_pts_2 = calc_mask.shape
 r2_mat = np.zeros_like(calc_mask).astype(float)*nan
 t_0 = time.time()

 for i_1 in range(n_pts_1-1, -1, -1):
  if 'print' in ftype and sqrt(n_pts_1 - i_1) % 1 == 0:
   print(f"{bps.size - i_1}/{bps.size} rows calculated, {(time.time() - t_0):.2f}")
  for i_2 in range(n_pts_2):
   if not calc_mask[i_1, i_2]:
    continue
   yr1, yr2 = bps_1[i_1, i_2], bps_2[i_1, i_2]  
   md = piece3fit(yrs, vals, yr1, yr2)
   r2_cur = r2_calc(yrs, vals, md)[1]
   r2_mat[i_1, i_2] = r2_cur
 
 if 'save' in ftype and vals_name != '':
  fn = f"{dir_r2}/{vals_name}.bin" if dir_r2 != '' else f"{vals_name}.bin"
  if dir_r2 not in os.listdir():
   os.mkdir(dir_r2)
  with open(fn, 'wb') as d:
   pickle.dump((r2_mat, bps), d)

 return r2_mat, bps

#load r2 matrix from drive (either by location and time series smooth window halfwidth, or direct file name)
def r2_load(loc='', hw=183, dir_r2='r2s', fn=''): #(+)
 vals_name = f"r2_{loc}_{hw}.bin" if fn == '' else fn
 if dir_r2 != '' and dir_r2 in os.listdir() and vals_name in os.listdir(dir_r2):
  with open(f"{dir_r2}/{vals_name}", 'rb') as d:
   r2_mat, bp_range_3 = pickle.load(d)
  return r2_mat, bp_range_3
 else:
  return arr_dummy, arr_dummy
  
#used in best solution search
#calculates array of R2 coefficient for a set of breakpoints (shape = (set length, 2))
def r2_list(yrs, vals, bps): # 5 raw 6 mth 7 half-year 8 year 9 2yr smoothed
 n_bp = bps.shape[0]
 r2s = np.zeros(n_bp)
 if bps.ndim == 1: #2-piecewise fit
  for i_bp in range(n_bp):
   yr_break = bps[i_bp]
   md = piece2fit_scipy(yrs, vals, yr_break=yr_break)
   r2_cur = r2_calc(yrs, vals, md)[1]
   r2s[i_bp] = r2_cur
 else:
  for i_bp in range(n_bp):
   yr1, yr2 = bps[i_bp]
   md = piece3fit(yrs, vals, yr1, yr2)
   r2_cur = r2_calc(yrs, vals, md)[1]
   r2s[i_bp] = r2_cur
 return r2s
  

#===== data visualization

#plots raw and smoothed anomalies and saves them to png according to f_type
#used for saving to png and interactive plotting
#accepts offsets on Y-axis
def anom_plot(ts=arr_dummy, vals=arr_dummy, loc=None, ftype='clf', title_str='', img_name='', dir_plot='', sm_type='mth half year 2yr', i_data=nan,
              base=(1960,1985), log_range=(-inf, inf), sm_stats=sm_stats, x_lims=None, y_lims=None, add_arg={}): #add yr_start, yr_end in f_type
 
 if loc != None:
  place = loc_name(loc)
  if not isfinite(i_data):
   weather_calc = isinstance(loc, int) or loc.isdigit()
   i_data = i_src_t_avg if 'Ice' not in loc else i_src_ice
  title_str = (((f"{loc} {place} temperature") if loc.isdigit() else loc) + f" anomalies (base: {base[0]} - {base[1]-1})") if title_str=='' else title_str
  ts, vals = t_log(loc, log_range=log_range, ftype='jds', i_data=i_data) #OK

 st = daily_stats(ts, vals, base=base)
 nds_st, val_avg = st[:,0].astype(int), st[:, i_st_avg]
 val_sm = daily_stats_smooth(val_avg, sm=sm_stats)
 dailystats = (st, val_sm)

 yr_color = 'yellow' if 'yr_color' not in add_arg.keys() else add_arg['yr_color'] #'#ff8020'
 lst = 'solid' if 'linestyle' not in add_arg.keys() else add_arg['linestyle']
 off_set = add_arg['offset'] if 'offset' in add_arg.keys() else 0  

 vs_out = {}
 st_str = '\nsmooth window widths: '
 if 'clf' in ftype:
  plt.close('all')
 if 'raw' in sm_type:
  anoms_raw  = anom_calc(ts, vals + off_set, dailystats=dailystats[0], sm_an=(-1,1,1), ftype='')
  plt.plot(jd_to_yr(ts), anoms_raw, linewidth=0.5, linestyle=lst,color ='#b0e0e0', alpha=0.5)
  vs_out['raw'] = anoms_raw
  st_str += ' none (cyan),'
 if '2mth' in sm_type:
  anoms_mth  = anom_calc(ts, vals + off_set, dailystats=dailystats[0], sm_an=(30,1,1), ftype='')
  plt.plot(jd_to_yr(ts)[15:-15], anoms_mth[15:-15], linewidth=1, linestyle=lst, color='#a0a0ff', alpha=0.9)
  vs_out['2mth'] = anoms_mth
  st_str += ' 2 mth (blue),'
 if 'year' in sm_type:
  anoms_half = anom_calc(ts, vals + off_set, dailystats=dailystats[0], sm_an=(183,1,1), ftype='')
  plt.plot(jd_to_yr(ts)[92:-92], anoms_half[92:-92], linewidth=1, linestyle=lst, color = 'green', alpha=0.8)
  vs_out['year'] = anoms_half
  st_str += ' year (green),'
 if '2yrs' in sm_type:
  anoms_1yr  = anom_calc(ts, vals + off_set, dailystats=dailystats[0], sm_an=(366,1,1), ftype='')
  plt.plot(jd_to_yr(ts)[183:-183], anoms_1yr[183:-183], linewidth=1, linestyle=lst, color = yr_color)
  vs_out['2yrs'] = anoms_1yr
  st_str += ' 2 yrs (yellow),'
 if '4yrs' in sm_type:
  anoms_2yr  = anom_calc(ts, vals + off_set, dailystats=dailystats[0], sm_an=(732,1,1), ftype='')
  plt.plot(jd_to_yr(ts)[366:-366], anoms_2yr[366:-366], linewidth=1, linestyle=lst, color = 'red', alpha=0.8)
  vs_out['4yrs'] = anoms_2yr
  st_str += ' 4 yrs (red)'
 plt.title(title_str + st_str) # plt.suptitle(suptitle_str, **{'x': 0.5, 'y': 0.05}) 
 if x_lims != None:
  plt.xlim(x_lims)
 if y_lims != None:
  plt.ylim(y_lims)

 plt.grid(True)
 
 if img_name != '':
  if dir_plot != '' and dir_plot not in os.listdir():
   os.mkdir(dir_plot)
  fn = dir_plot + '/' + img_name if dir_plot != '' else img_name
  plt.gcf().set_size_inches(13, 6)
  plt.savefig(fn, dpi=200)
  plt.close()
 else:   
  plt.ion()
  plt.show()
 
 return vs_out


#(+)
#plots piecewise and linear fits, writes r2, intercepts and slopes, saves a png
#takes best piecewise fits brom r2 matrices and lists loaded from .bin files
#to plot non-global-best 3pcwise fit, truncate r2-matrix manually around local optimum
#specify truncating conditions by lambda-function in **f_type of r2_load
#~1 minute per location with piece3fit_best_calc
def fit_compare(yrs=arr_dummy, vals=arr_dummy, loc='', bp_range=(1970,2015), bp_min_diff=0.2, title_str='', img_name='', dir_plot='', ftype='', 
                cond=None, dir_r2='r2s', r2_mat=arr_dummy, r2_fn='', log_range=(1960,2025), base=(1965, 1985), sm_anom=(183,1,1)):

 if loc != '':  
  weather_calc = loc.isdigit() 
  place = names[Blashyrkh.index(loc)]
  hw_an = sm_anom[0]
  title_str = (((f"{loc} {place} temperature ") if weather_calc else place + ' ') +  f"anomaly (base: {base[0]} - {base[1]})") if title_str=='' else title_str
  title_r2 = '\n R-squared vs break points'
  title_sm = f"; smooth window width:{hw_an*2} d"
  yrs_full, vals_full = t_log_calc(loc, log_range=log_range, base=base, sm_stats=sm_stats, sm_anom=sm_anom)[:2]
  yrs, vals = yrs_full[int(hw_an/3):-int(hw_an/3)], vals_full[int(hw_an/3):-int(hw_an/3)]
  bp_range = (max(yrs.min() + 2, bp_range[0]), min(yrs.max() - 2, bp_range[1]) )
  r2_fn = f"r2_{loc}_{hw_an}.bin" if r2_fn=='' else r2_fn
 
 plt.clf()
 plt.plot(yrs, vals) 
 #simple linear fit, two-piecewise fit, three-piece-wise fit
 lin_fit = np.polyfit(yrs, vals, 1, rcond=None, full=True, w=None, cov=False)
 md_1 = np.array([[-inf, lin_fit[0][0], lin_fit[0][1], inf],])
 r2_1 = r2_calc(yrs, vals, md_1)[1]
 bp, r2_best_2, md_2 = piece2fit_best_calc(yrs, vals, bp_range=bp_range, thr=0.3, n_pts=100, ctr_lim=10)
 if 'mat' in ftype:
  r2_mat, bps = r2_load(fn=r2_fn, dir_r2=dir_r2) 
  if r2_mat.size == 0:
   r2_mat, bps = r2_calc_all_3(yrs, vals, bp_range=bp_range, bp_min_diff=bp_min_diff,
                 res_yr = 0.1 if hw_an < 183 else 0.2, dir_r2=dir_r2)
  bp_1, bp_2, r2_best_3, md_3 = piece3fit_best_mat(yrs, vals, bp_range=bp_range, fn=r2_fn, cond=cond, ftype='')
 else:
  bp_1, bp_2, r2_best_3, md_3 = piece3fit_best_calc(yrs, vals, bp_range=bp_range, thr=0.3, bp_min_diff=bp_min_diff, cond=cond, n_pts=1000, ctr_lim=10)

#----------------plotting models------------
 piecewise_plot(md_1, lstyle=(1, 'dashed', '#00ff00'))
 piecewise_plot(md_2, lstyle=(1, 'dashed', '#00ff00'))
 piecewise_plot(md_3, lstyle=(2, 'dashed', '#ff8060'))
 
#-------------------text info-----------
 slope_3_1, slope_3_2, slope_3_3 = md_3[:,1]
 inter_3_1, inter_3_2, inter_3_3 = md_3[:,2]
 slope_2_1, slope_2_2 = md_2[:,1]
 inter_2_1, inter_2_2 = md_2[:,2]
 slope_lin, inter_lin = md_1[0,1], md_1[0,2]

 model_lin_str = f"linear: R2 adj = {r2_1*1000:.2f}\n {inter_lin:.0f} + {slope_lin:.3f} * yr" 
 model_2_str = f"2-spline:  R2 adj =  {r2_best_2*1000:.2f} \n break pt {bp:.2f}"
 model_3_r_str = f"3-spline: R2 adj = {r2_best_3*1000:.2f} \n breakpoints: {bp_1:.2f}, {bp_2:.2f}"
 model_3_l_str = f"  spline 1: {inter_3_1:.2f} + {slope_3_1:.3f} * yr\n  spline 2: {inter_3_2:.2f} + {slope_3_2:.3f} * yr\n spline 3: {inter_3_3:.2f} + {slope_3_3:.3f} * yr" 
 
 stat_str = f"slope 1 {slope_3_1:.3f} slope 2 {slope_3_2:.3f} slope 3 {slope_3_3:.3f}"
 
 plt.title(title_str)
 ice_calc = 'Ice' in title_str or 'Snow' in title_str # if graph is descending, put text info into the lower left corner
 (x_l, y_l) = (0.72, 0.14) if not ice_calc else (0.15, 0.14)
 plt.figtext(x_l, y_l, model_lin_str + '\n\n' + model_2_str + '\n\n' + model_3_r_str + '\n' + model_3_l_str)
 plt.axis('tight')
 plt.xlim(yrs.min(), yrs.max())
 y_min, y_max = vals.min(), vals.max()
 plt.ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))
 plt.ylabel( (title_str[:title_str.index(' anom')] + ' anomaly') if ice_calc else 'temperature anomaly')
 plt.grid(True)
 
 if img_name != '':
  if dir_plot != '' and dir_plot not in os.listdir():
   os.mkdir(dir_plot)
  fn = dir_plot + '/' + img_name if dir_plot != '' else img_name
  plt.gcf().set_size_inches(13, 6)
  plt.savefig(fn, dpi=200)
  plt.close()
 else:   
  plt.ion()
  plt.show()
 
 return None#{'r_corr_1': r_corr_1, 'model_1': model_lin, 'data_calc_1': data_calc_1, 'r_corr_2': r_corr_2, 'model_2': best_fit_2, 'data_calc_2': data_calc_2, 'r_corr_3': r_corr_3, 'model_3': best_fit_3, 'data_calc_3': data_calc_3, 'data_in': t_log}


#plots R2 values vs. break point 1 and break point 2 
#takes pre-calculated R@ matrix
def r2_plot(r2s, bp_range, piece2fits=arr_dummy, img_name='', dir_plot='', cmap='plasma', 
            z_lims=None, plot_lims=None, title_str='', ftype='', best_pt=(nan,nan,nan)): #OK
 if 'clf' in ftype:
  plt.close('all')
 pc2_present = piece2fits.size > 0
 
 bps_1, bps_2 = np.meshgrid(bp_range, bp_range, indexing='ij')
 fig, ax = plt.subplots()
 if z_lims == None:
  z_min, z_max = (np.nanmin(r2s), max(np.nanmax(r2s), np.nanmax(piece2fits[:,1]) if pc2_present else 0.0))
  clr_norm = colors.TwoSlopeNorm(vmin=0.0, vcenter=0.9*z_max, vmax=z_max)
 else:
  z_min, z_max = z_lims
  clr_norm = colors.Normalize(vmin=z_min, vmax=z_max)
 p = ax.pcolor(bps_1, bps_2, r2s, norm=clr_norm, cmap=cmap, shading='auto') 
 cb = fig.colorbar(p, ax=ax) #, extend='max')
 if plot_lims != None:
  ax.set_xlim(plot_lims)
  ax.set_ylim(plot_lims)
 if pc2_present: #
  yrs_2, r2_list = piece2fits.T
  i_max_2 = np.nanargmax(r2_list)
  yr_max_2, r2_max_2 = yrs_2[i_max_2], r2_list[i_max_2]
  plt.scatter(yrs_2, yrs_2, c=r2_list, norm=clr_norm, cmap=cmap, s=5)
  ax2 = fig.add_axes([0.45, 0.22, 0.25, 0.2]) #left, bottom, width, height #https://stackoverflow.com/questions/21001088/how-to-add-different-graphs-as-an-inset-in-another-python-graph
  ax2.scatter(yrs_2, r2_list, c=r2_list, norm=clr_norm, cmap=cmap, s=2)
  ax2.set_title('Single break point fit')
  ax2.grid(True)
 if best_pt == (nan, nan, nan): #indicate best point
  i_max_3pc = np.unravel_index(np.nanargmax(r2s), r2s.shape)
  y_max_3pc, x_max_3pc = bp_range[i_max_3pc[1]], bp_range[i_max_3pc[0]] #'ij' indexing
  r_max_3pc = r2s[i_max_3pc]
  x_max = yr_max_2 if pc2_present and r2_max_2 > r_max_3pc else x_max_3pc
  y_max = yr_max_2 if pc2_present and r2_max_2 > r_max_3pc else y_max_3pc
  r_max = r2_max_2 if pc2_present and r2_max_2 > r_max_3pc else r_max_3pc
 else:
  x_max, y_max, r_max = best_pt
 ax.scatter(x_max, y_max, c='#ff4040', s=5)

 descr_str = f"Best fit, 2 breakpoints: {x_max:.2f}, {y_max:.2f} , R2: {r_max:.3f}; 1 breakpoint: {yr_max_2:.2f}, R2: {r2_max_2:.3f}" #'Best fit, 2 breakpoints: ' + '%.2f' % x_max + ', '  + '%.2f' % y_max + ', R2: ' + '%.3f' % r_max + '; 1 breakpoint: ' + '%.2f' % yr_max_2 + ', R2: ' + '%.3f' % r2_max_2
 
 ax.set_title(title_str)
 ax.set_xlabel("Break point 1", fontsize=14)
 ax.set_ylabel("Break point 2", fontsize=14)
 ax.grid(True, linestyle=':', color= 'gray')
 fig.text(0.17, 0.13, descr_str, fontsize=10)
 
 if img_name != '':
  if dir_plot != '' and dir_plot not in os.listdir():
   os.mkdir(dir_plot)
  fn = dir_plot + '/' + img_name if dir_plot != '' else img_name
  plt.gcf().set_size_inches(11, 6)
  plt.savefig(fn, dpi=200)
  plt.close()
 else:   
  plt.ion()
  plt.show()

  return fig, ax, cb
 
#plots R2 values:vs. break points 1 and 2 (3pc-wise fit) and vs. single BP vs breakpoints in three-piece-wise-linear fit
#loads R2 matrix if found on disk, calculates other values
#takes location code
#presently only for average temperature or ice extent/volume series
def r2_viz(loc, bp_range=(1970,2015), sm_anom=(183,1,1), base=None, log_range=(1965,2020), bp_min_diff=1.0, r2_res=1.0, cond=None,
           piece2fits=arr_dummy, vals_name='', img_name='', dir_plot='', cmap='plasma', z_lims=None, plot_lims=None, title_str='', ftype='', best_pt=(nan,nan,nan)): #OK
 weather_calc = loc.isdigit() 
 place = loc_name(loc) 
 hw_an = sm_anom[0]
 base = ((1960, 1985) if weather_calc else (1981, 1994)) if base==None else base
 title_str = ((f"{loc} {place} temperature ") if weather_calc else place + ' ') +  f"anomaly (base: {base[0]} - {base[1]})"
 title_r2 = '\n R-squared vs break points'
 title_sm = f"; smooth window width:{hw_an*2} d"
 
 yrs_full, vals_full = t_log_calc(loc, log_range=log_range, base=base, sm_stats=sm_stats, sm_anom=sm_anom)
 yrs, vals = yrs_full[int(hw_an/3):-int(hw_an/3)], vals_full[int(hw_an/3):-int(hw_an/3)]
 
 r2_lst, bp_range_2 = r2_calc_all_2(yrs, vals, bp_range=bp_range, res_yr=r2_res, ftype='')
 pc2fits = np.vstack((bp_range_2, r2_lst)).T
 
 vals_name = img_name if vals_name == '' else vals_name
 r2_mat, bp_range_3 = r2_load(loc, hw=sm_anom[0])
 if r2_mat.size == 0 or 'recalc' in ftype:
  ft_calc = 'save' if 'save' in ftype or 'recalc' in ftype else ''
  r2_mat, bp_range_3 = r2_calc_all_3(yrs, vals, bp_range=bp_range, bp_min_diff=bp_min_diff, res_yr=r2_res, ftype=ft_calc, vals_name=vals_name) 

 if cond != None:
  bps_1, bps_2 = np.meshgrid(bp_range_3, bp_range_3, indexing='ij')
  mask_r2 = np.where(cond(bps_1, bps_2), False, True)
  r2_v = np.ma.masked_array(r2_mat, mask=mask_r2, fill_value=nan)
 else:
  r2_v = r2_mat
 
 if 'save' in ftype and vals_name != '':
  fn = f"{dir_plot}/{vals_name}.bin" if dir_plot != '' else f"{vals_name}.bin"
  if dir_plot not in os.listdir():
   os.mkdir(dir_plot)
  with open(fn, 'wb') as d:
   pickle.dump((r2_mat, bp_range_3), d)
  
 plts = r2_plot(r2_v, bp_range_3, piece2fits=pc2fits, z_lims=z_lims, plot_lims=plot_lims, img_name=img_name, dir_plot=dir_plot, 
               ftype='clf', title_str=title_str+title_r2+title_sm, cmap='plasma')
 
 return r2_v, bp_range_3, r2_lst, bp_range_2, plts

#visualization of climatic oscillations, i.e. ao = Arctic oscillation
#'.ao', '.aao', '.na', '.pna'
def osc_viz(osc_type, x_lims=None, y_lims=None, ftype=''):
 osc_dir = 'Osc_inds'
 fns = os.listdir(osc_dir)
 fn = osc_dir + '/' + fns[ [osc_type in x for x in fns].index(True)]

 osc_0 = np.genfromtxt(fn, delimiter=',', skip_header=1)

 osc_raw = np.zeros((osc_0.shape[0], 2))
 osc_raw[:,0] = jd_to_yr(ymd_to_jd(osc_0[:,:3]) )
 osc_raw[:,1] = osc_0[:,-1]
 yrs, osc = osc_raw.T

 osc_sm0 = smoother(xs=yrs, ys=osc, hw=0.08, deg=1, n_iter=1)
 osc_sm1 = smoother(xs=yrs, ys=osc, hw=0.5, deg=1, n_iter=1)
 osc_sm2 = smoother(xs=yrs, ys=osc, hw=1, deg=1, n_iter=1)
 osc_sm3 = smoother(xs=yrs, ys=osc, hw=2, deg=1, n_iter=1)

 plt.ion()
 plt.show()
 plt.clf()
 plt.plot(yrs, osc, linewidth=0.5, linestyle='solid',color ='#b0e0e0', alpha=0.7)
 plt.plot(yrs, osc_sm0, linewidth=1, linestyle='solid', color='blue', alpha=0.3)
 plt.plot(yrs, osc_sm1, linewidth=1, linestyle='solid', color = 'green', alpha=0.9)
 plt.plot(yrs, osc_sm2, linewidth=1, linestyle='solid', color = 'orange')
 plt.plot(yrs, osc_sm3, linewidth=1, linestyle='solid', color = 'red')
 plt.grid(True)

 if x_lims != None:
  plt.xlim(x_lims)

 if y_lims != None:
  plt.ylim(y_lims)

 if 'save' in ftype:
  plt.gcf().set_size_inches(13, 6)
  plt.savefig(f"Osc_inds/Osc_{osc_type[1:]}.png", dpi=200)
 
 return yrs, osc
 
#plots climatogram for chosen location, reference period, and smoothing type.
#output format for t_stats: [0] mins, [1] avgs, [2] maxs, [3] precip.
#for each sub-list:
#[0] -  daynum, [1] mth [2] day [3]  avg [4] min [5] year of min [6] max [7] year of max [8] rsd

def climatogram(loc, base=(1965,1980), sm=sm_stats, t_scale=None, rsd_scale=None, ftype='clf', dir_plot='climatograms', dir_txt='climatograms_txt', prob_xtr=nan, sm_xtr=(5,1,3)):
 loc_name = names[Blashyrkh.index(loc)]
 loc_src = src_dir + '/' + str(loc) + '.csv'
 tbl = data_load(loc_src)
 climate_stats = stats_temp(tbl, base=base, sm=sm) #T_min_st, T_avg_st, T_max_st, prec_st =
 T_min, T_avg, T_max, prec_avg = climate_stats[:,:, i_st_avg]
 T_minmin, T_maxmax = climate_stats[0, :, i_st_min], climate_stats[2, :, i_st_max]
 T_rsd = climate_stats[1, :, i_st_rsd]
 T_min_sm, T_avg_sm, T_max_sm, prec_sm, T_rsd_sm = [ daily_stats_smooth(x, sm=sm) for x in [T_min, T_avg, T_max, prec_avg, T_rsd]]
 nds = np.arange(1, 367)
 
 if not isnan(prob_xtr):
  T_xtr_min, T_xtr_max = t_xtr_calc(loc, prob=prob_xtr, base=base, hw_distr=5, sm_prob=sm_xtr, sm_fin=(12,1,3), ftype='')
  T_minmin_sm, T_maxmax_sm = T_xtr_min[:,0],T_xtr_max[:,0]
 else:
  T_minmin_sm, T_maxmax_sm = [ daily_stats_smooth(x, sm=sm) for x in [T_minmin, T_maxmax]]
 ylims1 = (T_minmin.min() - 5, T_maxmax.max() + 5) if t_scale == None else t_scale # not in f_type.keys() else f_type['t_scale']
 ylims2 = (-0.05 * max(prec_avg.max(), T_rsd.max()), 1.2 * max(prec_avg.max(), T_rsd.max())) if rsd_scale == None else rsd_scale # not in f_type.keys() else f_type['rsd_scale']
 if 'clf' in ftype:
  plt.close('all')
 fig, ax1 = plt.subplots()
 ax1.plot(nds, T_min_sm, linewidth=2, color='#8080ff')
 ax1.plot(nds, T_avg_sm, linewidth=2, color='#20a020')
 ax1.plot(nds, T_max_sm, linewidth=2, color='#ff8040')
 ax1.plot(nds, T_maxmax_sm, linewidth=2, linestyle='dotted', color='#e00000')
 ax1.plot(nds, T_minmin_sm, linewidth=2, linestyle='dotted', color='#0000e0')
 ax1.plot(nds, T_min, color='#8080ff', alpha=0.2, marker='^', linewidth=1, markersize=2)
 ax1.plot(nds, T_avg, color='#40c040', alpha=0.2, marker='^', linewidth=1, markersize=2)
 ax1.plot(nds, T_max, color='#ff8040', alpha=0.2, marker='^', linewidth=1, markersize=2)
 ax1.plot(nds, T_maxmax, color='#d04020', alpha=0.2, marker='^', linewidth=1, markersize=2)
 ax1.plot(nds, T_minmin, color='#4020d0', alpha=0.2, marker='^', linewidth=1, markersize=2)
 ax2 = ax1.twinx()
 ax2.plot(nds, prec_sm, linewidth=3, color='#b0d0ff')
 ax2.plot(nds, T_rsd_sm, linewidth=3, linestyle='dashed', color='#80b080')
 yr_start, yr_end = int(max(base[0], tbl[0,1])), base[1]-1
 title_string_1 = f"{loc} {loc_name} climatogram base {yr_start} - {yr_end}\n"
 title_string_2 = f"halfwidth: {sm[0]} poly: {sm[1]} n_iter: {sm[2]}"
 title_string = title_string_1 + title_string_2
 plt.title(title_string)
 plt.xlim(1, 366)
 ax1.set_ylim(ylims1)
 ax2.set_ylim(ylims2)
 t_ticks = np.arange(ceil(ylims1[0]/10)*10, ceil(ylims1[1]/10)*10, 10)
 ax1.set_yticks(t_ticks)
 ax1.grid(visible=True, which='both', axis='y')
 d0 = 1
 for i in range(len(m_lengths)):
  plt.axvline(d0, color='gray')
  ax2.text(d0 + 10, (0.98*ylims2[0] + 0.02*ylims2[1]), m_names[i], fontsize=12)
  d0 += m_lengths[i]
 day_nums = np.tile(np.arange(0, 31, 3), 12) + np.repeat(m_len_sums, 11)
 for dn in day_nums:
  vl = plt.axvline(dn, lw=0.5, color='gray', alpha=0.2)

 yl1, yl2 = int(ylims1[0]), int(ylims1[-1]) + 1
 temp_marks = np.arange(yl1, yl2, 1)
 for t_mark in temp_marks:
  hl = ax1.axhline(t_mark, lw=0.5, color='gray', alpha=0.2)
 
 vals_all = np.vstack((nds, T_min, T_avg, T_max, T_min_sm, T_avg_sm, T_max_sm, T_minmin, T_maxmax, T_minmin_sm, T_maxmax_sm, prec_sm, T_rsd_sm)).T
 
 if 'save' in ftype:
  if dir_plot not in os.listdir():
   os.mkdir(dir_plot)
  img_name = f"{dir_plot}/{loc}_{loc_name}_{yr_start}_{yr_end-1}_climatogram.png"
  plt.gcf().set_size_inches(11, 6)
  plt.savefig(img_name, dpi=200)
  plt.close()
 else:
  plt.ion()
  plt.show()
 
 if 'txt' in ftype:
  head_str = '#mth day   T_min T_avg T_max T_minmin T_maxmax  T_rsd  prec_avg\n'
  if dir_txt not in os.listdir():
   os.mkdir(dir_txt)
  txt_name = f"{dir_txt}/{loc}_{loc_name}_{yr_start}_{yr_end-1}_climatogram.txt"
  str_out = ''
  for i_str in range(vals_all.shape[0]):
   vals_cur = vals_all[i_str]
   m, d = daynum_to_md(i_str+1)
   if d == 1:
    str_out += ('\n' if m != 1 else '') + head_str
   Tmin, Tavg, Tmax, Tminsm, Tavgsm, Tmaxsm, Tminmin, Tmaxmax, Tminminsm, Tmaxmaxsm, precsm, Trsdsm = vals_cur[1:]
   str_cur = f"{m:<2}{d:>3}{Tminsm:9.2f}{Tavgsm:7.2f}{Tmaxsm:7.2f}{Tminmin:8.1f}{Tmaxmax:6.1f}{Trsdsm:9.2f}{precsm:9.2f}\n"
   str_out += str_cur
  with open(txt_name, 'w') as f:
   f.write(str_out)
 return vals_all


#makes all plots, specified above, for a single location 
def plot_loc(loc, sm_type='year', base=None, sm_anom=(183,1,1), bp_range=(1970,2015), log_range=(1960,2025), 
             r2_res=0.5, bp_min_diff=0.25, ftype='anom climatogram fitcomp r2', dir_plot='', dir_plot_custom={}):
 
 weather_calc = loc.isdigit() 
 place = names[Blashyrkh.index(loc)]
 hw_an = sm_anom[0]
 base = ((1960, 1985) if not weather_calc else (1981, 1994)) if base==None else base
 
 yrs_full, vals_full, vs_raw, t_stats = t_log_calc(loc, log_range=log_range, base=base, sm_stats=sm_stats, sm_anom=sm_anom, ftype='full')
 yrs, vals = yrs_full[int(hw_an/3):-int(hw_an/3)], vals_full[int(hw_an/3):-int(hw_an/3)]
 ts = yr_to_jd(yrs_full)

 title_str = ((f"{loc} {place} temperature ") if weather_calc else place + ' ') +  f"anomaly (base: {base[0]} - {base[1]})"
 title_r2 = '\n R-squared vs break points'
 title_sm = f"; smooth window width:{hw_an*2} d"

 if 'clim' in ftype and weather_calc:
  dp_clim = dir_plot_custom['clim'] if 'clim' in dir_plot_custom.keys() else 'climatograms'
  tbl_0 = data_load(src_dir+'/'+ loc +'.csv')  
  vs_cl_full = climatogram(loc, base=(-inf, inf), sm=sm_stats)
  nds, T_min, T_avg, T_max, T_min_sm, T_avg_sm, T_max_sm, T_minmin, T_maxmax, T_minmin_sm, T_maxmax_sm, prec_sm, rsd_sm = vs_cl_full.T
  t_min, t_max = np.nanmin(T_minmin), np.nanmax(T_maxmax)
  t_scale = floor(t_min / 10) * 10.0, ceil(t_max/10.0) * 10.0
  ax2_max = max(np.nanmax(prec_sm), np.nanmax(rsd_sm))
  rsd_scale = (-0.05 * ax2_max, 1.2 * ax2_max)
  vs_cl_1 = climatogram(loc, base=(1965, 1981), sm=sm_stats, t_scale=t_scale, rsd_scale=rsd_scale, dir_plot=dp_clim, ftype='save')
  vs_cl_2 = climatogram(loc, base=(2008, 2024), sm=sm_stats, t_scale=t_scale, rsd_scale=rsd_scale, dir_plot=dp_clim, ftype='save')
  vs_cl_3 = climatogram(loc, base=(-inf, 1981), sm=sm_stats, t_scale=t_scale, rsd_scale=rsd_scale, dir_plot=dp_clim, ftype='save')
 if 'anom' in ftype:
  dp_anom = dir_plot_custom['anom'] if 'anom' in dir_plot_custom.keys() else 'anom plots'
  anoms = anom_plot(ts, vs_raw, base=base, ftype='clf', img_name=f"anoms_{loc}", dir_plot=dp_anom, title_str=title_str, sm_type='2mth year 2yrs 4yrs', x_lims=None, y_lims=None, add_arg={})
 if 'fitcomp' in ftype:
  dp_fc = dir_plot_custom['fitcomp'] if 'fitcomp' in dir_plot_custom.keys() else 'fits'
  fit_compare(yrs, vals, bp_range=bp_range, title_str=title_str + title_sm, img_name=f"fits_{loc}_{hw_an}", dir_plot=dp_fc, ftype='plot', bp_min_diff=bp_min_diff)

 if 'r2' in ftype:
  dp_r2 = dir_plot_custom['r2s'] if 'r2s' in dir_plot_custom.keys() else 'r2s'
  r2_lst, bp_range_2 = r2_calc_all_2(yrs, vals, bp_range=bp_range, res_yr=r2_res, ftype='')
  pc2fits = np.vstack((bp_range_2, r2_lst)).T
  r2_mat, bp_range_3 = r2_calc_all_3(yrs, vals, bp_range=bp_range, bp_min_diff=bp_min_diff, res_yr=r2_res, ftype='')
  plts = r2_plot(r2_mat, bp_range_3, piece2fits=pc2fits, img_name=f"r2s_{loc}_{hw_an}", dir_plot=dp_r2, ftype='clf', title_str=title_str+title_r2+title_sm, cmap='plasma')
 return None



#def t_anom_spread_single(loc, ts=arr_dummy, vals=arr_dummy, sm_stats=sm_stats, hw=5/366):
 

#=====supplementary

#returns smoothing window half-width for "smooth code", i.e. '1yr' - 183
def smooth_hw(sm_type=''):
 sm_type_det = [x in sm_type for x in sm_types]
 i_smt = sm_type_det.index(True) if any(sm_type_det) else 0
 smt = sm_types[i_smt] 
 hw = smooth_hws[i_smt]
 return hw, smt

#takes location code, returns location name   
def loc_name(loc):
 if isinstance(loc, int):
  loc = str(loc)
 return names[Blashyrkh.index(loc)] if loc in Blashyrkh else ''

#takes location code, returns filename containing source csv data   
def loc_to_filename(loc):
 fn = loc if (isinstance(loc, str) and  '.' in loc) else (src_dir+ '/' + str(loc) + '.csv')
 return fn


def t_distr_single(m, d, ymds, vals, daily_avgs, hw=10, sm_probs=(5, 1, 3), ftype=''): 
 daynum_0 = md_to_daynum(m, d)
 n_pts = ymds.shape[0]
 i_match_lst, anom_weights_lst = [], []
 weights_0 = (hw + 1 - abs(np.arange(-hw, hw+1))) if 'tri' in ftype else (gauss(0, hw/3, np.arange(-hw, hw+1)) if 'gauss' in ftype else np.ones(2*hw+1))
 weights = weights_0 / weights_0.sum()
 for i_shift in range(-hw, hw+1):
  daynum = ((daynum_0 + i_shift) - 1) % 366 + 1
  m_sh, d_sh = daynum_to_md(daynum)
  i_match_ymd = np.where((ymds[:,1] == m_sh) * (ymds[:,2] == d_sh))[0]
  i_match_lst.append(i_match_ymd)
  anom_weights_lst.append(np.ones_like(i_match_ymd).astype(float) * weights[i_shift]) #triangular window, normalized later
 i_match = np.hstack(i_match_lst) 
 anom_weights = np.hstack(anom_weights_lst) 
 anoms = vals[i_match]
 anom_rsd = anoms.std()
 T_avg = daily_avgs[daynum-1]
 Ts = anoms + T_avg #detrended temperatures
 
 Ts_un, prob_int = prob_int_calc(Ts, ws=anom_weights, close_thr=0.1)
 return Ts_un, prob_int

def t_xtr_calc(loc, prob=0.01, base=(1965,1985), hw_distr=10, sm_prob=(5,1,3), sm_fin=(12,1,3), ftype=''):
 probs = (prob,) if isinstance(prob, float) else prob
 n_xtrs = len(probs)
 jds, t_anoms_min, ts_min, daily_avgs_min = t_log_calc(loc, log_range=base, base=base, sm_stats=sm_stats, sm_anom=None, i_data=i_src_t_min, ftype='full jds')
 jds, t_anoms_max, ts_max, daily_avgs_max = t_log_calc(loc, log_range=base, base=base, sm_stats=sm_stats, sm_anom=None, i_data=i_src_t_max, ftype='full jds')
 ymds = jd_to_ymd(jds)
 xtrs_min = np.zeros((366, n_xtrs))*nan
 xtrs_max = np.zeros((366, n_xtrs))*nan
 for i_d in range(366):
  m, d = daynum_to_md(i_d)
  Ts_min_md, prob_int_min_md = t_distr_single(m, d, ymds, t_anoms_min, daily_avgs_min, hw=hw_distr, ftype=ftype)
  Ts_max_md, prob_int_max_md = t_distr_single(m, d, ymds, t_anoms_max, daily_avgs_max, hw=hw_distr, ftype=ftype)
  Ts_sm_min, pi_min = prob_int_smooth(Ts_min_md, prob_int_min_md, sm_probs=sm_prob)
  Ts_sm_max, pi_max = prob_int_smooth(Ts_max_md, prob_int_max_md, sm_probs=sm_prob)
  for i_p in range(len(probs)):
   xtrs_min[i_d, i_p] = prob_solve(Ts_sm_min, pi_min, probs[i_p])
   xtrs_max[i_d, i_p] = prob_solve(Ts_sm_max, pi_max, 1-probs[i_p])
 
 xtrs_min_f, xtrs_max_f = np.copy(xtrs_min), np.copy(xtrs_max)
 if sm_fin != None and sm_fin[0] > 1:
  for i_p in range(len(probs)):
   xtrs_min_f[:,i_p] = daily_stats_smooth(xtrs_min[:,i_p], sm_fin)
   xtrs_max_f[:,i_p] = daily_stats_smooth(xtrs_max[:,i_p], sm_fin)
 return xtrs_min_f, xtrs_max_f
