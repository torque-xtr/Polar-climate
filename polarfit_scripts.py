#data analysis and plotting scripts

from polarfit import *

#=====data extraction

#-----temperature and precipitation log extraction #tested OK

for loc in Blashyrkh:
 frost = weather_extract(loc)

#-----ice log extraction #tested OK

#copied from module ice_parser.py
#https://nsidc.org/arcticseaicenews/sea-ice-tools/

src_lst = []
os.chdir('source_ice')
with open('NSIDC_daily_src.csv', 'r') as srccsv:
 lines = srccsv.readlines()
 for i in range(1, len(lines)):
  new_l = lines[i].strip('\n').split('\t') 
  new_line = [float(x) if x.replace('.', '').isdecimal() else 'ND' for x in new_l][:48] #data up to [47] which is 2023
  src_lst.append(new_line)

src_lst_trans = [[y[i] for y in src_lst] for i in range(48)]

#[0][] - month, [1][] - day, n_column - year.
lin_lst = []
day_ctr = 43873
for i in range(3, 48):
 for j in range(366):
  yr = i + 1976
  mth = src_lst_trans[0][j]
  day = src_lst_trans[1][j]
  if not (not isleap(yr) and mth == 2 and day == 29):
   day_ctr += 1
   yr_float = mjd_to_yr_s(day_ctr)
   new_line = [yr_float, src_lst_trans[i][j], day_ctr, yr, mth, day]
   lin_lst.append(new_line)


with open('ice_data.csv', 'w') as icedata:
 for i in range(len(lin_lst)):
  csv_line = '%.3f' % lin_lst[i][0] + '\t' + str(lin_lst[i][1]) + '\t' + str(int(lin_lst[i][2])) + '\t'+ str(int(lin_lst[i][3])) + '\t' + str(int(lin_lst[i][4])) + '\t' + str(int(lin_lst[i][5])) + '\n'
  wrt = icedata.write(csv_line)


os.chdir('..')

#=====data process

#writing smoothed logs to csv
for loc in Blashyrkh:  #tested OK
 if loc == 'ice':
  t_an = anom_ice('ice_data.csv', 1981, 1994)
 else:
  t_an, t_log, t_st, t_st_sm = anom_temps(str(loc) + '.csv', 1965, 1985)
 ftype_smth = {'halfwidths': [0.08, 0.5, 1, 2], 'raw': True, 'loc': loc} #write if needed
 t_sm = anom_writer(t_an, **ftype_smth) # 5 raw 6 mth 7 half-year 8 year 9 2yr smoothed


#-----two-piece-wise fit
#-----build and save R2 vs breakpoint lists
#tested OK
t_0 = time.time()
ctr = 0
for loc in Blashyrkh:
 for smt in sm_types:
  ctr += 1
  r2_list = p2fit_calc(loc, 1000, **{'sm_type': smt, 'savebin': True, 'draw': True, 'save': True})
  print(ctr, loc, smt, time.time() - t_0)

#-----three-piece-wise fit
#-----build and save R2 vs breakpoints matrices

t_0 = time.time()
ctr = 0
for loc in Blashyrkh: #tested OK
 for smt in sm_types:
  ctr += 1
  r_matrix = r2_calc(loc, 15000, 1965, 2022, 1965, 2022, **{'sm_type': smt, 'prefix': ''}) #'raw_save' in older versions
  print(ctr, loc, smt, time.time() - t_0)


#=====plotting

plt.ion()
plt.draw()

#---r matrices

#full
#tested OK
#f_type_r2plot['trunc'] = (lambda x, y: not (x < 1985 and y > 2015)) #cut upper left rectangle
f_type_r2plot = {'colorcode': 'blues', 'base': 0, 'pow': 6, 'rel': True, 'rel_scale': 1.0, 'show': False, 'save': True}
t_0 = time.time()
for loc in Blashyrkh:
 for smt in sm_types:
  rmtr, rmt = r2_plot(loc, smt, **f_type_r2plot)
  print(loc, smt, time.time() - t_0)

#truncated
#tested OK
#(lambda x, y: not (x < 1985 and y > 2015))
#(lambda x, y: not (x < 1985 and y > 2010))

f_type_r2plot = {'colorcode': 'blues', 'base': 0, 'pow': 6, 'rel': True, 'rel_scale': 1.0, 'show': False, 'save': True, 'plot_lim': 'default'}
f_type_r2plot['trunc'] = (lambda x, y: x > 1995 and x < 2010 and y > x + 2.5 and y < 2017)
t_0 = time.time()
for loc in Blashyrkh:
 for smt in sm_types:
  rmtr, rmt = r2_plot(loc, smt, **f_type_r2plot)
  print(loc, smt, time.time() - t_0)


#---climatograms

for yr_start in range(1963, 2014): #tested OK
 climatogram(27612, yr_start, yr_start+10, **{'hw': 10, 't_scale': (-40, 40), 'rsd_scale': (0, 8), 'save': True})

#---rows
for loc in Blashyrkh: #tested OK
 for yr_start in range(1963, 2014):
  climatogram(20087, yr_start, yr_start+10, **{'hw': 10, 't_scale': (-50, 15), 'rsd_scale': (0, 9), 'save': True, 'dpi':200})

climatogram(27612, 2007, 2023, **{'hw': 10, 't_scale': (-35, 45), 'rsd_scale': (0, 7.5), 'save': True, 'dpi':400})

#---anomaly plots

#general
f_type = {'sm_type': 'mth half year 2yr', 'x_lims': (1925, 2023), 'y_lims': (-5, 7), 'yr_color': 'yellow'} #default '#ff8020'
for loc in Blashyrkh: #[20107, 21824, 23226, 24959, 25563, 21946, 25248]:
 plt_anom = anom_plot(loc, **f_type, **{'img_prefix': 'general', 'save': True}) #correct spelling for new naming

#zoomed on 1990 - 2020
f_type = {'sm_type': 'mth half year 2yr', 'x_lims': (1990, 2020), 'y_lims': (-3.5, 9.5), 'yr_color': 'yellow'}
for loc in Blashyrkh: #[20107, 21824, 23226, 24959, 25563, 21946, 25248]:
 plt_anom = anom_plot(loc, **f_type, **{'img_prefix': 'interest', 'save': True})   #correct spelling for new naming

#zoomed on 1990 - 2020
f_type = {'sm_type': 'mth half', 'x_lims': (2000, 2008), 'y_lims': (-5, 9), 'yr_color': 'yellow'}
for loc in Blashyrkh: #[20107, 21824, 23226, 24959, 25563, 21946, 25248]:
 plt_anom = anom_plot(loc, **f_type, **{'img_prefix': 'mega', 'save': True}) 

#abrupt warming in Kara sea region
f_type = {'sm_type': 'mth half', 'x_lims': (1987, 2023), 'y_lims': (-4.5, 8.5)}
for loc in [20069, 20046, 20087, 20292, 20891, 20674]:
 plt_anom = anom_plot(loc, **f_type, **{'img_prefix': 'step_Kara', 'save': True}) 

#2004 step on half-year-smoothed logs 
f_type = {'sm_type': 'mth half', 'x_lims': (1998, 2008), 'y_lims': (-4.5, 8.5)}
for loc in [20107, 20069, 20087, 20292, 20674, 20891, 21824, 21982, 23226]:
 plt_anom = anom_plot(loc, **f_type, **{'img_prefix': 'step_Kara__2', 'save': True}) 

#transition in eastern locations
f_type = {'sm_type': 'mth half', 'x_lims': (1975, 2023), 'y_lims': (-6, 7)}
for loc in [25051, 25282, 25248]:
 plt_anom = anom_plot(loc, **f_type, **{'img_prefix': 'step_Eastern', 'save': True}) 

#transition in continental locations
f_type = {'sm_type': 'mth half', 'x_lims': (1987, 2023), 'y_lims': (-4, 7)}
for loc in [24266, 24688, 24959]:
 plt_anom = anom_plot(loc, **f_type, **{'img_prefix': 'step_Sib', 'save': True}) 

#--interactives
plt_anom = anom_plot('ice', **{'sm_type': 'raw mth half yr', 'save': False, 'plot': True})

#steps using y-axis offsets
plt.clf()
off_set = 0
for loc in [20107, 20087, 20292, 21824, 21982]:
 plt_anom = anom_plot(loc, **{'sm_type': 'mth half', 'save': False, 'plot': True, 'offset': off_set})
 off_set += 5

plt.clf()
off_set = 0
for loc in [20107, 20087, 20069, 20292, 20674, 20891, 21824]:
 plt_anom = anom_plot(loc, **{'sm_type': 'half', 'save': False, 'plot': True, 'offset': off_set})
 off_set += 2.5


plt.clf()
off_set = 0
for loc in [26063, 27612, 22113, 24266]:
 plt_anom = anom_plot(loc, **{'sm_type': 'mth', 'save': False, 'plot': True, 'offset': off_set})
 off_set += 5



#---fit plots #tested OK

#full 
t_0 = time.time()
f_type_fcs = {'plot': False, 'save': True, 'r2 plot': False, 'trunc_2': (lambda x: x < 2021 and x > 1975)}
for loc in Blashyrkh: # [20107, 21824, 23226, 24959, 25563, 21946, 25248]:
 for smt in sm_types:
  out_data = fit_compare_simple(loc, smt, **f_type_fcs)
  print(loc, smt, time.time() - t_0)
   
#truncated r_matrices
t_0 = time.time()
f_type_fcs = {'plot': False, 'save': True, 'r2 plot': False, 'trunc_2': (lambda x: x < 2021 and x > 1975), 'suffix': 'inter'}
for loc in Blashyrkh: 
 for smt in sm_types:
  out_data = fit_compare_simple(loc, smt, **f_type_fcs, **{'trunc_3': (lambda x, y: x > 1995 and x < 2010 and y > x + 0.01 and y < 2017)})
  print(loc, smt, time.time() - t_0)

#step search:

t_0 = time.time()
f_type_fcs = {'plot': False, 'save': True, 'r2 plot': False, 'trunc_2': (lambda x: x < 2021 and x > 1975), 'suffix': 'inter'}
for loc in [20046, 20069, 20087, 20107, 20292, 'ice']: 
 for smt in sm_types:
  out_data = fit_compare_simple(loc, smt, **f_type_fcs, **{'trunc_3': (lambda x, y: x > 1995 and x < 2010 and y < x + 2 and y < 2017)})
  print(loc, smt, time.time() - t_0)


#=====writing logs

#-----other csvs

#calculate linear fits from 1965 to 2000 and from 2007 to present, and jump values between them
all_models = [] #OK
t_0 = time.time()
for loc in Blashyrkh:
 fits_st = fit_compare_linear_stats(loc, 'mth')
 all_models.append(fits_st)
 print(loc, time.time() - t_0)

popped = all_models.pop(len(all_models)-1)
all_models.sort(key=lambda x: x[5])
all_models.append(popped)
with open('jump_fits.csv', 'w') as csv:
 wrt = csv.write('loc\tk1\tk1 rsd\tb1\tb1 rsd\tk2\tk2 rsd\tb2\tb2 rsd\tjump\tj_rsd\tval1\tv1rsd\tval2\tv2_rsd\n\n')
 for i in range(len(all_models)):
  nstr = all_models[i] 
  loc = nstr[5]
  wrt = csv.write(str(loc) + '\t' + '%.4f' % nstr[0][1] + '\t' + '%.4f' % nstr[0][0]  + '\t' + '%.2f' % nstr[1][1] + '\t' + '%.2f' % nstr[1][0]  + '\t' + '%.4f' % nstr[2][1] + '\t' + '%.4f' % nstr[2][0]  + '\t' + '%.2f' % nstr[3][1] + '\t' + '%.2f' % nstr[3][0] + '\t' + '%.2f' % nstr[4][1]  + '\t' + '%.2f' % nstr[4][0]+ '\t' + '%.2f' % nstr[6][1]  + '\t' + '%.2f' % nstr[6][0]+ '\t' + '%.2f' % nstr[7][1]  + '\t' + '%.2f' % nstr[7][0] + '\n')


#---2pc and 3pc-wise models summary
#---not so good, use previous script

ctr = 0
t_0 = time.time()
model_data_list = []
f_type_fcs = {'plot': False, 'save': False, 'r2 plot': False, 'trunc_2': (lambda x: x < 2021 and x > 1975), 'trunc_3': (lambda x, y: x > 1995 and x < 2010 and y > x + 2.5 and y < 2017)}
for loc in Blashyrkh:
 for sm_type in sm_types:
  if loc != 'ice':
   continue
  out_data = fit_compare_simple(loc, **f_type_fcs)
  r2_1 = out_data['r_corr_1']
  r2_2 = out_data['r_corr_2']
  r2_3 = out_data['r_corr_3']
  slope_1 = out_data['model_1'][0][1]
  slope_2_1 = out_data['model_2'][0][1]
  bp_2_1 = out_data['model_2'][0][3]
  slope_2_2 = out_data['model_2'][1][1]
  slope_3_1 = out_data['model_3'][0][1]
  bp_3_1 = out_data['model_3'][0][3]
  slope_3_2 = out_data['model_3'][1][1]
  bp_3_2 = out_data['model_3'][1][3]
  slope_3_3 = out_data['model_3'][2][1]
  fin_pt_1 = 2023*slope_1 + out_data['model_1'][0][2]
  fin_pt_2 = 2023*slope_2_2 + out_data['model_2'][1][2]
  fin_pt_3 = 2023*slope_3_3 + out_data['model_3'][2][2]
  tbl_entry = {'loc': loc, 'sm_type': sm_type, 'r2_1': r2_1, 'r2_2': r2_2, 'r2_3': r2_3, 'slope_1': slope_1, 'slope_2_1': slope_2_1, 'bp_2_1': bp_2_1, 'slope_2_2': slope_2_2, 'slope_3_1': slope_3_1, 'bp3_1': bp_3_1, 'slope_3_2': slope_3_2, 'bp3_2': bp_3_2, 'slope_3_3': slope_3_3, 'bp_3_2': bp_3_2, 'fin_pt_1': fin_pt_1, 'fin_pt_2': fin_pt_2, 'fin_pt_3': fin_pt_3}
  model_data_list.append(tbl_entry)
  ctr += 1
  print(ctr, loc, sm_type, time.time() - t_0)


#---manual_fit:

#loc: i.e. 20069 for Wiese island or 'ice' for ice extent
model_3 = manual_fit(loc, **{'sm_type': 'half', 'l_cut': 1965})
