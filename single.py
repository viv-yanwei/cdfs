#!/bin/sh
import numpy as np
from astropy.io import fits
import os
import os.path
import math
import random as rd
import matplotlib.pyplot as plt

from ciao_contrib.runtool import *
from sherpa.astro.ui import *
from sherpa_contrib.profiles import *

full = fits.open("/Users/yan/Box/CDFS/cdfs.fits")
#
cid = full[1].data["XID_SOURCE_NUMBER"]
zbest = full[1].data["REDSHIFT"]
photon_index = full[1].data["PHOTON_INDEX"]

#fflux =  full[1].data["CDWFS_FLUX_F"]

czf = zip(cid,zbest,photon_index)

czf_z=[] # 1st cut - redshift:
         # select those with normal redshift (aka z>0)
for x in czf:
    if x[1]>0:
        czf_z.append(x)   # 3 col: xid, z, photon_index, ne, pe






file=open("/Users/yan/Box/CDFS/1119/selected_xid.txt","r")
lines=file.readlines()
xid_sel=[]
for x in lines:
    xid_sel.append(x[:-1])
file.close()


redo=[]
fit_gamma_c=[]
#input_gamma=[]
for i in range(167,169):#len(czf_z)):#400
    
    xid = str(czf_z[i][0])	

    
    if xid in xid_sel:
        print("yes")
    #else:
        
        xid_dir = "/Users/yan/Box/CDFS/1119/source/"+xid
        print(i)
           
        if os.path.isdir(xid_dir) == True:
            os.chdir(xid_dir)
        
            
            load_data("1119_combined_src.pi")
            load_bkg("1119_combined_bkg.pi")


            if os.path.isfile("spec_combined_src.arf") == True:
                load_arf("spec_combined_src.arf")
            else:
                redo.append(xid)
	
            if os.path.isfile("spec_combined_src.rmf") == True:
                load_rmf("spec_combined_src.rmf")
            else:
                redo.append(xid)
	
            if os.path.isfile("spec_combined_bkg.arf") == True:
                load_bkg_arf("spec_combined_bkg.arf")
            
            else:
                redo.append(xid)
        
            if os.path.isfile("spec_combined_bkg.rmf") == True:
                load_bkg_rmf("spec_combined_bkg.rmf")
            else:
                redo.append(xid)


            z=czf_z[i][1]
            photon=czf_z[i][2]

 
            create_model_component("powlaw1d","p")
            create_model_component("xszphabs", "q")    
            set_source(powlaw1d.p*xszphabs.q)
   
	#show_model()
    
            thaw(p.gamma)
            p.gamma.min = -5
            p.gamma.max = 5
            p.gamma = photon
            freeze(p.gamma)
            q.redshift=z
    

    
            notice(1,10)
            ignore(6/(1+z),7/(1+z))



    


            set_stat("cstat")
            #set_stat("chi2gehrels")
            fit()
    
#    freeze()

#    set_stat("chi2gehrels")
      
            #group_counts(1,20)  
    
            subtract() 
#    fit()
    
            
    
            
            notice(1,10)
            group_counts(1,20) 
            plot_fit()
            #plot_data()
            #r=get_ratio_plot()
            #save_arrays("ratio_cstat.txt", [r.x,r.y,r.xerr,r.yerr], ["x","y","xerr","yerr"],clobber="TRUE")

            
            #fg = get_par("p.gamma")            
            #fgval = fg.val
             
            #with open("gamma_fit.txt", "w") as f:
            #     f.write(str(fgval) +"\n")
            #f.close()

            #fit_gamma_c.append(fgval)
            #input_gamma.append(photon)
    #print(i)
    
    i=i+1
 
with open("redo.txt", "w") as f:
	for s in redo:
    		f.write(str(s) +"\n")
f.close()
 
# with open("gamma_out_chi.txt", "w") as f:
# 	for s in fit_gamma:
#     		f.write(str(s) +"\n")
# f.close()


 
# with open("gamma_inp_chi.txt", "w") as f:
# 	for s in input_gamma:
#     		f.write(str(s) +"\n")
# f.close()