#data analysis and plotting scripts

from polarfit import *

#=====data extraction

#-----temperature and precipitation log extraction 
#(+)
quit()
python
from polarfit import *

for loc in Blashyrkh:
 wx = weather_extract(loc)

#=======================plotting===============================

#-------------------- all charts----------------------------

#takes ~ 2 minutes per location
#(+)
quit()
python
from polarfit import *

t_0 = time.time()
for loc in [x for x in Blashyrkh if x.isdigit()]: #isdigit: weather, not isdigit: ice and other data; True: all
 print(f"{time.time()-t_0:.2f} plotting {loc} ({loc_name(loc)}) ...")
 plt.close('all')
 log_range, bp_range, base = ((1965,2025), (1970,2015), (1960,1985)) if loc.isdigit() else ((1981,2025), (1990,2020), (1985,2000))
 plot_loc(loc, base=base, sm_anom=(366,1,1), bp_range=bp_range, log_range=log_range, r2_res=1.0, ftype='anom climatogram fitcomp')
 plot_loc(loc, base=base, sm_anom=(732,1,1), bp_range=bp_range, log_range=log_range, r2_res=1.0, ftype='fitcomp')
 plot_loc(loc, base=base, sm_anom=(183,1,1), bp_range=bp_range, log_range=log_range, r2_res=0.5, ftype='r2 fitcomp')
 plot_loc(loc, base=base, sm_anom=(30,1,1), bp_range=bp_range, log_range=log_range, r2_res=0.2, ftype='r2 fitcomp')


#---------------- climatograms---------------------------

#(+)
quit()
python
from polarfit import *

dp_clim, dp_txt = 'climatograms', 'climatograms_txt'
t_0 = time.time()
for loc in [x for x in Blashyrkh if x.isdigit()]:
 print(f"{time.time()-t_0:.2f} {loc} ({loc_name(loc)})")
 plt.close('all')
 vs_cl_full = climatogram(loc, base=(-inf, inf), sm=sm_stats, ftype='')
 nds, T_min, T_avg, T_max, T_min_sm, T_avg_sm, T_max_sm, T_minmin, T_maxmax, T_minmin_sm, T_maxmax_sm, prec_sm, rsd_sm = vs_cl_full.T
 t_min, t_max = np.nanmin(T_minmin), np.nanmax(T_maxmax)
 t_scale = floor(t_min / 10) * 10.0, ceil(t_max/10.0) * 10.0
 ax2_max = max(np.nanmax(prec_sm), np.nanmax(rsd_sm))
 rsd_scale = (-0.05 * ax2_max, 1.2 * ax2_max)
 vs_cl_1 = climatogram(loc, base=(1965, 1981), sm=sm_stats, t_scale=t_scale, rsd_scale=rsd_scale, ftype='save txt', dir_plot=dp_clim, dir_txt=dp_txt, prob_xtr=0.01, sm_xtr=(5,1,3))
 vs_cl_2 = climatogram(loc, base=(2009, 2025), sm=sm_stats, t_scale=t_scale, rsd_scale=rsd_scale, ftype='save txt', dir_plot=dp_clim, dir_txt=dp_txt, prob_xtr=0.01, sm_xtr=(5,1,3))
 vs_cl_3 = climatogram(loc, base=(-inf, 1981), sm=sm_stats, t_scale=t_scale, rsd_scale=rsd_scale, ftype='save txt', dir_plot=dp_clim, dir_txt=dp_txt, prob_xtr=0.01, sm_xtr=(5,1,3))


#--------------------anomalies---------------------

#=====plotting special

#---climatogram rows #(+)

loc='20069'
vs_cl_full = climatogram(loc, base=(-inf, inf), sm=sm_stats)
nds, T_min, T_avg, T_max, T_min_sm, T_avg_sm, T_max_sm, T_minmin, T_maxmax, T_minmin_sm, T_maxmax_sm, prec_sm, rsd_sm = vs_cl_full.T
t_min, t_max = np.nanmin(T_minmin), np.nanmax(T_maxmax)
t_scale = floor(t_min / 10) * 10.0, ceil(t_max/10.0) * 10.0
ax2_max = max(np.nanmax(prec_sm), np.nanmax(rsd_sm))
rsd_scale = (-0.05 * ax2_max, 1.2 * ax2_max)

for yr_start in range(1963, 2015):
 vs_cl = climatogram(loc, base=(yr_start, yr_start+11), sm=sm_stats, t_scale=t_scale, rsd_scale=rsd_scale, dir_plot=f"climatogram_rows_{loc}", ftype='save')

#---------anomaly plots


#title string is created from loc in function
quit()
python
from polarfit import *

#(+)
dp='anom plots'
for loc in Blashyrkh:
 log_range = (1965,2025)
 base = (1965, 1980) if loc.isdigit() else (1981, 1994)
 fn = f"anoms_{loc}"
 vs=anom_plot(loc=loc, ftype='clf', img_name=fn, dir_plot=dp, sm_type='2mth year 2yrs 4yrs', base=base, i_data=nan,
              sm_stats=sm_stats, log_range=log_range, x_lims=None, y_lims=None)

#(+)
dp='anom plots new base'
for loc in Blashyrkh:
 log_range = (1965,2025)
 base = (2009, 2025)
 fn = f"anoms_{loc}"
 vs=anom_plot(loc=loc, ftype='clf', img_name=fn, dir_plot=dp, sm_type='2mth year 2yrs 4yrs', base=base, i_data=nan,
              sm_stats=sm_stats, log_range=log_range, x_lims=None, y_lims=None)


#zoomed on 1990 - 2020

quit()
python
from polarfit import *

#(+)
dp='anom plots_1990_2020'
for loc in Blashyrkh:
 log_range = (1965,2025)
 base = (1965, 1980) if loc.isdigit() else (1981, 1994)
 fn = f"90_20_anoms_{loc}"
 vs=anom_plot(loc=loc, ftype='clf', img_name=fn, dir_plot=dp, 
              sm_type=('2mth year' if loc.isdigit() else 'raw 2mth year'), base=base, i_data=nan,
              sm_stats=sm_stats, log_range=log_range, x_lims=(1990,2020), 
              y_lims=((-6,10) if loc.isdigit() else None))


#abrupt warming in Kara sea region
#(+)
dp='anom plots_spec'
for loc in ['20069', '20046', '20087', '20292', '20891', '20674']:
 log_range, base = (1965,2025), (1965, 1980)
 fn = f"step_Kara_anoms_{loc}"
 vs=anom_plot(loc=loc, ftype='clf', img_name=fn, dir_plot=dp, sm_type='2mth year', base=base, i_data=nan,
              sm_stats=sm_stats, log_range=log_range, x_lims=(1987,2025), y_lims=(-4.5,8.5))


#2004 step, zoomed
#(+)
dp='anom plots_spec'
for loc in ['20107', '20069', '20087', '20292', '20674', '20891', '21824', '21982', '23226']:
 log_range, base = (1965,2025), (1965, 1980)
 fn = f"step_Kara_2_anoms_{loc}"
 vs=anom_plot(loc=loc, ftype='clf', img_name=fn, dir_plot=dp, sm_type='2mth year', base=base, i_data=nan,
              sm_stats=sm_stats, log_range=log_range, x_lims=(1998,2008), y_lims=(-4.5,8.5))


#transition in eastern locations
#(+)
dp='anom plots_spec'
for loc in ['25051', '25282', '25248']:
 log_range, base = (1965,2025), (1965, 1980)
 fn = f"step_Eastern_anoms_{loc}"
 vs=anom_plot(loc=loc, ftype='clf', img_name=fn, dir_plot=dp, sm_type='2mth year', base=base, i_data=nan,
              sm_stats=sm_stats, log_range=log_range, x_lims=(1975,2025), y_lims=(-6, 7))


#transition in continental locations
#(+)
dp='anom plots_spec'
for loc in ['24266', '24688', '24959']:
 log_range, base = (1965,2025), (1965, 1980)
 fn = f"step_Sib_anoms_{loc}"
 vs=anom_plot(loc=loc, ftype='clf', img_name=fn, dir_plot=dp, sm_type='2mth year', base=base, i_data=nan,
              sm_stats=sm_stats, log_range=log_range, x_lims=(1987,2025), y_lims=(-4, 7))


#------------------------------------


#steps using y-axis offsets

quit()
python
from polarfit import *

#(+)
dp='anom plots_spec'
off_set = 0.0
locs = ['20107', '20087', '20292', '21824', '21982']
t_str = 'Temperature anomalies:\n' + ' ,'.join([names[Blashyrkh.index(x)] for x in locs])
for loc in locs:
 add_args = {'offset': off_set}
 log_range, base = (1965,2025), (1965, 1980)
 fn = "step_Kara_all"
 vs=anom_plot(loc=loc, ftype='', img_name='' if loc != locs[-1] else fn, dir_plot=dp, title_str=t_str, sm_type='2mth year', base=base,
              sm_stats=sm_stats, log_range=log_range, x_lims=(1998,2008), y_lims=None, add_arg=add_args)
 off_set += 5.0


#(+)
dp='anom plots_spec'
off_set = 0.0
locs = ['20107', '20087', '20069', '20292', '20674', '20891', '21824']
t_str = 'Temperature anomalies:\n' + ' ,'.join([names[Blashyrkh.index(x)] for x in locs])
for loc in locs:
 add_args = {'offset': off_set}
 log_range, base = (1965,2025), (1965, 1980)
 fn = "step_Kara_all"
 vs=anom_plot(loc=loc, ftype='', img_name='' if loc != locs[-1] else fn, dir_plot=dp, title_str=t_str, sm_type='2mth year', base=base,
              sm_stats=sm_stats, log_range=log_range, x_lims=(2002,2008), y_lims=None, add_arg=add_args)
 off_set += 5.0


#(+)
dp='anom plots_spec'
off_set = 0.0
locs = ['26063', '27612', '22113', '24266']
t_str = 'Temperature anomalies:\n' + ' ,'.join([names[Blashyrkh.index(x)] for x in locs])
for loc in locs:
 add_args = {'offset': off_set}
 log_range, base = (-inf,inf), (1965, 1980)
 fn = "1940s"
 vs=anom_plot(loc=loc, ftype='', img_name='' if loc != locs[-1] else fn, dir_plot=dp, title_str=t_str, sm_type='2mth', base=base,
              sm_stats=sm_stats, log_range=log_range, x_lims=(1935,1955), y_lims=None, add_arg=add_args)
 off_set += 5.0




#----------------------r-squared vs breakpoints

#calculate r2 arrays, 2m and year smoothing

quit()
python
from polarfit import *

#(+)
#400 s per location if r2 matrix is calculated
anom_smoothing = (183, 1, 1) #window half-width, polynomial degree, number of passes)
t_0 = time.time()
for loc in Blashyrkh:
 print(f"{(time.time() - t_0):.2f} calculating {loc}" )
 bp_range, log_range = ((1970,2020), (1965,2025)) if loc.isdigit() else ((1985,2020), (1980,2025))
 vs=r2_viz(loc, bp_range=bp_range, sm_anom=(183,1,1), base=None, cond=None, log_range=log_range,  
           bp_min_diff=0.2, r2_res=0.2, piece2fits=arr_dummy, img_name=f"r2_{loc}_183", dir_plot='r2s', cmap='seismic', 
           z_lims=None, plot_lims=None, title_str='', ftype='save recalc', best_pt=(nan,nan,nan))
 vs=r2_viz(loc, bp_range=bp_range, sm_anom=(30,1,1), base=None, cond=None, log_range=log_range,  
           bp_min_diff=0.1, r2_res=0.1, piece2fits=arr_dummy, img_name=f"r2_{loc}_30", dir_plot='r2s', cmap='seismic', 
           z_lims=None, plot_lims=None, title_str='', ftype='save recalc', best_pt=(nan,nan,nan))


#visualize 'interesting area' for locations with erratic data on the edges
quit()
python
from polarfit import *

cond = lambda x, y: inrange_arr(x, (1980, 2015)) * inrange_arr(y, (1980, 2015))
anom_smoothing = (183, 1, 1) #window half-width, polynomial degree, number of passes)
t_0 = time.time()
for loc in ['20107', '21982', '22113', '22550', '23589', '25051', '25248', '25282', '25399', '25563', '27164', '28440', '29638',  '34391', '37461']: #Blashyrkh:
 print(f"{(time.time() - t_0):.2f} calculating {loc}" )
 bp_range, log_range = ((1970,2020), (1965,2025)) if loc.isdigit() else ((1985,2020), (1980,2025))
 vs=r2_viz(loc, bp_range=bp_range, sm_anom=(183,1,1), base=None, cond=cond, log_range=log_range,  
           bp_min_diff=0.2, r2_res=0.2, piece2fits=arr_dummy, img_name=f"r2_{loc}_183_mid", dir_plot='r2s', cmap='seismic', 
           z_lims=None, plot_lims=None, title_str='', ftype='', best_pt=(nan,nan,nan))
 vs=r2_viz(loc, bp_range=bp_range, sm_anom=(30,1,1), base=None, cond=cond, log_range=log_range,  
           bp_min_diff=0.1, r2_res=0.1, piece2fits=arr_dummy, img_name=f"r2_{loc}_30_mid", dir_plot='r2s', cmap='seismic', 
           z_lims=None, plot_lims=None, title_str='', ftype='', best_pt=(nan,nan,nan))



#---fit plots (+)

#full 
quit()
python
from polarfit import *

t_0 = time.time()
for loc in Blashyrkh: # [20107, 21824, 23226, 24959, 25563, 21946, 25248]:
 fit_compare(loc=loc, bp_range=(1965,2025), bp_min_diff=0.2, title_str='', img_name=f"fits_{loc}", dir_plot='fit_compare', ftype='', 
             cond=None, dir_r2='r2s', r2_mat=arr_dummy, r2_fn='', log_range=(1960,2025), base=(1965, 1985), sm_anom=(183,1,1)) #ftype='mat': load r2 matrix
 print(loc, time.time() - t_0)

   
#----------------selected ranges of break points

#-----middle_range

quit()
python
from polarfit import *

cond = lambda x, y: (x > 1990) * (x < 2015) * (y < 2017)
t_0 = time.time()
for loc in Blashyrkh: # [20107, 21824, 23226, 24959, 25563, 21946, 25248]:
 fit_compare(loc=loc, bp_range=(1970,2015), bp_min_diff=0.0, title_str='', img_name=f"fits_{loc}_mid", dir_plot='fit_compare', ftype='', 
             cond=cond, dir_r2='r2s', r2_mat=arr_dummy, r2_fn='', log_range=(1960,2025), base=(1965, 1985)) #ftype='mat': load r2 matrix
 print(loc, time.time() - t_0)


#---------step search

quit()
python
from polarfit import *

cond = lambda x, y: inrange_arr(y - x, (0.1, 3.0))
t_0 = time.time()
for loc in Blashyrkh: # [20107, 21824, 23226, 24959, 25563, 21946, 25248]:
 fit_compare(loc=loc, bp_range=(1970,2015), bp_min_diff=0.0, title_str='', img_name=f"fits_{loc}_step", dir_plot='fit_compare', ftype='', 
             cond=cond, dir_r2='r2s', r2_mat=arr_dummy, r2_fn='', log_range=(1960,2025), base=(1965, 1985)) #ftype='mat': load r2 matrix
 print(loc, time.time() - t_0)


#---manual fit (+)

quit()
python
from polarfit import *

loc = '20069'
base = (1960, 1985)
log_range = (1963, 2025)
sm_stats = (12,1,3)
sm_anom = (183, 1, 1)
ts, Ts = t_log_calc(loc, log_range=log_range, base=base, sm_stats=sm_stats, sm_anom=sm_anom, i_data=-1, ftype='1yr')

md = manual_fit(ts, Ts, ftype='1yr save')


#--- climate oscillation index visualizations--------- #(+)

#https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii
#https://ftp.cpc.ncep.noaa.gov/cwlinks/

quit()
python
from polarfit import *

for osc_type in ['.ao', '.aao', '.na', '.pna']:
 yrs, osc = osc_viz(osc_type, x_lims=(1960,2025), y_lims=None, ftype='save')

yrs, osc = osc_viz('.ao')

