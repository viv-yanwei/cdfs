#!/usr/bin/env python2
# -*- coding: utf-8 -*-
  
import numpy as np

import os
import os.path

import math

from scipy.interpolate import interp1d
import matplotlib.pyplot as pl
from astropy.io import fits

from astropy import stats as sts



from scipy.integrate import quad
from scipy.optimize import curve_fit

import matplotlib as mpl
from scipy.stats import chisquare

from matplotlib.backends.backend_pdf import PdfPages

 
from pylab import *
from matplotlib import gridspec


   
mpl.rc('font', family='sans-serif') 
mpl.rc('font',family='Times New Roman')
mpl.rc('text', usetex='false') 

fig = pl.figure(figsize=(10,10))


#########################################

#fig = plt.figure()

   

ax = [pl.subplot(4,2,i+1) for i in range(4)]

for a in ax:
    #a.set_yticklabels(["netflux"])
   # a.set_ylabel('Netflux',fontsize=15)
    #a.set_xlabel('Energy', fontsize=15)
    if a==ax[1] or a==ax[3]:
        a.yaxis.tick_right()        
    for tick in a.xaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
        tick.label1.set_fontweight('bold')#'bold
    for tick in a.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)             ############### Save thick axes
        tick.label2.set_fontsize(18)
        tick.label1.set_fontweight('bold')#
        tick.label2.set_fontweight('bold')#
    #a.set_yticklabels(np.arange(0,10,5), fontsize=15)
    
    #tick.label1.set_fontweight()
   # a.set_yticklabels(np.arange(0,10,5), fontsize=20)
    #a.set_ylim(0, 18)

pl.subplots_adjust(wspace=0, hspace=0)


p_lir_nox={}
p_lir_nox[0]=np.array([0.5, 0.1, 0.05, 0.02])
p_lir_nox[1]=np.array([0.9, 0.5, 0.2, 0.05])



f = fits.open("/Users/yan/Box/CDFS/cdfs.fits")
name = f[1].data['XID_SOURCE_NUMBER']
fhb = f[1].data['HB_FLUX']
nh = f[1].data['NH']
br = f[1].data['BAND_RATIO']
chb = f[1].data['HB_COUNTS']
lx = f[1].data['RF_LX']




file=open("/Users/yan/Box/CDFS/1119/brl_nhl.lis","r")
lines=file.readlines()
xid=[]
for x in lines:
    xid.append(x[:-1])
file.close()

num_ll=len(xid)


file=open("/Users/yan/Box/CDFS/1119/brl_nhh.lis","r")
lines=file.readlines()
for x in lines:
    xid.append(x[:-1])
file.close()

num_lh=len(xid)-num_ll


file=open("/Users/yan/Box/CDFS/1119/brh_nhl.lis","r")
lines=file.readlines()
for x in lines:
    xid.append(x[:-1])
file.close()

num_hl=len(xid)-num_lh-num_ll

file=open("/Users/yan/Box/CDFS/1119/brh_nhh.lis","r")
lines=file.readlines()
for x in lines:
    xid.append(x[:-1])
file.close()

num_hh=len(xid)-num_hl-num_lh-num_ll



ener = np.arange(0.075,15.0,0.15)
main_dr='/Users/yan/Box/CDFS/1119/source'

#xid=xid[1:222]
array_bn=np.append(ener,ener)

 
num=0
fhb_new=[]
br_new=[]
nh_new=[]
name_new=[]
chb_new=[]
lx_new=[]

for j in range(0,len(xid)):
#for i in range(0,len(xid)):
    os.chdir('/Users/yan/Box/CDFS/1119/source/'+xid[j])
    #load_pha("1119_combined_src.pi")
    #load_arf("spec_combined_src.arf")
    #load_rmf("spec_combined_src.rmf")
    #load_bkg("1119_combined_bkg.pi")
    # if os.path.isfile("spec_combined_bkg.arf") == True:
    #     load_bkg_arf("spec_combined_bkg.arf")
    # if os.path.isfile("spec_combined_bkg.arf") == True:
    #     load_bkg_rmf("spec_combined_bkg.rmf")
    #subtract()


    #group_counts(1,20)
    #create_model_component("powlaw1d","mdl")

    f=open("z.txt","r")
    lines=f.readlines()
    z=float(lines[0][:-1])

    #notice(1,10)
    #ignore(6/(1+z),7/(1+z))
    #set_source(powlaw1d.p)
    ##show_model()
    #p.gamma.min = 1
    #p.gamma.max = 2
    #p.gamma = 1.4
    #fit()
    #plot_fit()
    #notice(1,10)
    #plot_data()
    #r=get_ratio_plot()
    #save_arrays("ratio.firs", [r.x, r.y, r.err, r.yerr], ["x", "y", "xerr", "yerr"], clobber = "TRUE")
     
     
    if os.path.isfile("./ratio.txt") == True: ## change directory ##
        #f = fits.open('./ratio.fits')
        f = open('./ratio.txt')
        lines=f.readlines()
        ener_obs = []
        flux_obs = []
        ener_obs_err = []
        flux_obs_err = []
        
        for x in lines[2:]:
            ener_obs.append(x.split(' ')[0])
            flux_obs.append(x.split(' ')[1])
            ener_obs_err.append(x.split(' ')[2])
            flux_obs_err.append(x.split(' ')[3][:-1])
            
            
        ener_obs = [float(i) for i in ener_obs] 
        flux_obs = [float(i) for i in flux_obs] 
        ener_obs_err = [float(i) for i in ener_obs_err] 
        flux_obs_err = [float(i) for i in flux_obs_err] 
        
        ener_res = np.asarray(ener_obs)*(1+z)
        flux_res = np.asarray(flux_obs)
        ener_res_err = np.asarray(ener_obs_err)*(1+z)
        flux_res_err = np.asarray(flux_obs_err)
        
        if len(flux_obs) > 3:
            func_res = interp1d(ener_res, flux_res, kind='cubic',bounds_error=False, fill_value=0)
        
            ener_bn = ener
            flux_bn = func_res(ener_bn)
            
            
            bsize_bn = np.diff(ener_bn)
            bsize_bn = np.append(bsize_bn,0)        
            fsize = interp1d(ener_bn, bsize_bn, kind='linear',bounds_error=False, fill_value=0)        
            bsize = fsize(ener)
            
            func_res_err = interp1d(ener_res, flux_res_err, kind='linear',
                                bounds_error=False, fill_value=0)
            
            
            err = func_res_err(ener)       
            flux_bn_err = err*np.sqrt(bsize/0.15)

        
            
            flux_bn = np.append(flux_bn, flux_bn_err)

            array_bn = np.vstack((array_bn, flux_bn))
            
            x=np.where(name == float(xid[j]))[0][0]
            nh_new.append(nh[x])
            fhb_new.append(fhb[x])
            br_new.append(br[x])
            name_new.append(name[x])
            chb_new.append(chb[x])
            lx_new.append(lx[x])

        
        num=num+1
        
        
        
    else:
        #print(xid_spec[i])
        print(j)
        
    j=j+1


os.chdir(main_dr)

np.save("brnh", array_bn)
np.save("fhb_new", fhb_new)
np.save("br_new", br_new)
np.save("nh_new", nh_new)
np.save("chb_new", chb_new)
np.save("lx_new", lx_new)


y_lir_nox=[]
area_lir_nox=[]
width_lir_nox=[]

p1_lir_nox=[]
reda=[]


p_lir_nox={}
p_lir_nox[0]=np.array([0.5, 0.1, 0.05, 0.02,6.4])
p_lir_nox[1]=np.array([0.9, 0.5, 0.2, 0.05,6.4])





def func(x, a, b, c, d,e):
#def func(x, a, b, c, d):
    return c*np.exp(-0.5*((x-e)/d)**2)+a*x**b


f =fits.open('/Users/yan/Box/CDFS/1119/f_bl_nl.fits')
xdata = f[0].data[0][32:55]
y =f[0].data[1][32:55]
y_noise = f[0].data[2][32:55]



p=p_lir_nox[0]
ydata = y + y_noise

popt, pcov = curve_fit(func, xdata, ydata, p0=p,
                       bounds=((-np.inf, -np.inf, 0, 0,6.4), 
                               (np.inf, np.inf, np.inf, 0.2,6.7)))

fungau = popt[2]*np.exp(-0.5*((xdata-popt[4])/(popt[3]))**2)
funpl = popt[0]*xdata**popt[1]#+popt[4]*xdata**popt[5]
funmod = fungau+funpl


ax[0].plot(xdata, fungau, "b--", linewidth=1)
ax[0].plot(xdata, funmod, "-", color="purple",linewidth=4,label="Best Fits")
ax[0].errorbar(xdata, y, yerr=y_noise,fmt='.', ecolor="blue",capsize=4, markersize=10,label="Net Flux")
ax[0].set_ylim(0,2)
ax[0].set_xlim(5,8)
def integrand(x):
    return popt[2]*np.exp(-0.5*((x-6.4)/popt[3])**2)
ans, err = quad(integrand, 5., 8.)
y_line=popt[0]*6.4**popt[1]#+popt[4]*6.4**popt[5]
y_lir_nox.append(y_line)
area_lir_nox.append(ans)
width_lir_nox.append(ans/y_line)    
print popt
print width_lir_nox




f =fits.open('/Users/yan/Box/CDFS/1119/bh_nl.fits')
xdata = f[0].data[0][32:55]
y =f[0].data[1][32:55]
y_noise = f[0].data[2][32:55]


p=p_lir_nox[0]
ydata = y + y_noise

popt, pcov = curve_fit(func, xdata, ydata, p0=p,
                       bounds=((-np.inf, -np.inf, 0, 0), 
                               (np.inf, np.inf, np.inf, 0.4)))

fungau = popt[2]*np.exp(-0.5*((xdata-6.4)/(popt[3]))**2)
funpl = popt[0]*xdata**popt[1]#+popt[4]*xdata**popt[5]
funmod = fungau+funpl


ax[1].plot(xdata, fungau, "b--", linewidth=1)
ax[1].plot(xdata, funmod, "-", color="purple",linewidth=4,label="Best Fits")
ax[1].errorbar(xdata, y, yerr=y_noise,fmt='.', ecolor="blue",capsize=4, markersize=10,label="Net Flux")
ax[1].set_ylim(0,2)
ax[1].set_xlim(5,8)
def integrand(x):
    return popt[2]*np.exp(-0.5*((x-6.4)/popt[3])**2)
ans, err = quad(integrand, 5., 8.)
y_line=popt[0]*6.4**popt[1]#+popt[4]*6.4**popt[5]
y_lir_nox.append(y_line)
area_lir_nox.append(ans)
width_lir_nox.append(ans/y_line)    
print popt
print width_lir_nox






f =fits.open('/Users/yan/Box/CDFS/1119/bl_nh.fits')
xdata = f[0].data[0][32:55]
y =f[0].data[1][32:55]
y_noise = f[0].data[2][32:55]


p=p_lir_nox[0]
ydata = y + y_noise

popt, pcov = curve_fit(func, xdata, ydata, p0=p,
                       bounds=((-np.inf, -np.inf, 0, 0), 
                               (np.inf, np.inf, np.inf, 0.4 )))

fungau = popt[2]*np.exp(-0.5*((xdata-6.4)/(popt[3]))**2)
funpl = popt[0]*xdata**popt[1]#+popt[4]*xdata**popt[5]
funmod = fungau+funpl


ax[2].plot(xdata, fungau, "b--", linewidth=1)
ax[2].plot(xdata, funmod, "-", color="purple",linewidth=4,label="Best Fits")
ax[2].errorbar(xdata, y, yerr=y_noise,fmt='.', ecolor="blue",capsize=4, markersize=10,label="Net Flux")
ax[2].set_ylim(0,2)
ax[2].set_xlim(5,8)
def integrand(x):
    return popt[2]*np.exp(-0.5*((x-6.4)/popt[3])**2)
ans, err = quad(integrand, 5., 8.)
y_line=popt[0]*6.4**popt[1]#+popt[4]*6.4**popt[5]
y_lir_nox.append(y_line)
area_lir_nox.append(ans)
width_lir_nox.append(ans/y_line)    
print popt
print width_lir_nox







f =fits.open('/Users/yan/Box/CDFS/1119/bh_nh.fits')
xdata = f[0].data[0][32:55]
y =f[0].data[1][32:55]
y_noise = f[0].data[2][32:55]


p=p_lir_nox[0]
ydata = y + y_noise

popt, pcov = curve_fit(func, xdata, ydata, p0=p,
                       bounds=((-np.inf, -np.inf, 0, 0), 
                               (np.inf, np.inf, np.inf, 0.4)))

fungau = popt[2]*np.exp(-0.5*((xdata-6.4)/(popt[3]))**2)
funpl = popt[0]*xdata**popt[1]#+popt[4]*xdata**popt[5]
funmod = fungau+funpl


ax[3].plot(xdata, fungau, "b--", linewidth=1)
ax[3].plot(xdata, funmod, "-", color="purple",linewidth=4,label="Best Fits")
ax[3].errorbar(xdata, y, yerr=y_noise,fmt='.', ecolor="blue",capsize=4, markersize=10,label="Net Flux")
ax[3].set_ylim(0,2)
ax[3].set_xlim(5,8)
def integrand(x):
    return popt[2]*np.exp(-0.5*((x-6.4)/popt[3])**2)
ans, err = quad(integrand, 5., 8.)
y_line=popt[0]*6.4**popt[1]#+popt[4]*6.4**popt[5]
y_lir_nox.append(y_line)
area_lir_nox.append(ans)
width_lir_nox.append(ans/y_line)    
print popt
print width_lir_nox
#############################################################################################################################################################################################################



#array_ratio = np.load('array_ratio.npy')
