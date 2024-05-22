from astropy.io import fits 
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import numpy as np
import pickle
from proj_functions import *
import scipy.constants as constant
import sys
import config
#from pandas import *
 

def cont_determ (wv, sp, remove_n_first, radius_min, radius_max, max_vicinity, \
                stretching, use_RIC, interp, use_denoise, usefilter, nu, niter_peaks_remove, denoising_distance, fwhm): 
    #data = read_csv("./spectra/case_2_spec.csv")
 
    # converting column data to list
    wavelengths = wv
    spectra = sp

#inputfile=sys.argv[1]

#with fits.open(inputfile) as hdu:

#    full_data = hdu[1].data
#    header = hdu[0].header

#wavelengths = full_data["wavelength"]
#spectra = full_data["flux"]
#uncertainties = full_data["error"]   

    min_lambda = min(wavelengths)
#FWHM=header['HIERARCH ESO QC CCF FWHM']   #FWHM in Km/s                            
#FWHM_WL= min_lambda*(FWHM/(constant.c/1000)) #FWHM in A
    FWHM_WL=fwhm #in case the spectra doesn't include information about FWHM

#plt.plot(wavelengths, spectra)
#plt.show()
#---------------------------------
#Parameters (change values in config.py)
#---------------------------------
    #remove_n_first=config.remove_n_first      #removes the first n points in the spectra (only if first points are not useful!) otherwise set to 0
    #radius_min=config.radius_min              #min alpha shape radius
    #radius_max=config.radius_max           #max alpha shape radius (should be at least the size of the largest gap)
    #max_vicinity = config.max_vicinity        #required number of values between adjacent maxima
    global_stretch =stretching      #normalization parameter
    use_pmap = use_RIC  #use the RIC to avoid large "dips" in the continuum (see documentation)
    #interp=config.interp    #interpolation type
    #use_denoise=config.use_denoise  #for noisy spectra use the average of the flux value around the maximum
    #usefilter=config.usefilter   #use savgol filter to smooth spectra 
    #nu=config.nu                    #exponent of the computed penalty (see documentation)                  
    #niter_peaks_remove=config.niter_peaks_remove  #number of iterations to remove sharpest peaks before interpolation
    #denoising_distance=config.denoising_distance   #number of points to calculate the average around a maximum if use_denoise is True, useful for noisy spectra

#---------------------------------
#print("Running...")
#-----------Smoothing------------------------------------------
    spectra_clip=spectra[remove_n_first:] #removes first n points
    wavelengths_clip=wavelengths[remove_n_first:]                                                                              
    spectra_clip, wavelengths_clip=smooth.rolling_sigma_clip(spectra_clip, wavelengths_clip, 20)     #sigma clip twice
    spectra_clip, wavelengths_clip=smooth.rolling_sigma_clip(spectra_clip, wavelengths_clip, 20)

    if(usefilter):
        spectra_clip = savgol_filter(spectra_clip, window_length=11, polyorder=3)     
 
    s1 = p_map.rolling_max(spectra_clip, wavelengths_clip, FWHM_WL*40)
    s2 = p_map.rolling_max(spectra_clip, wavelengths_clip, FWHM_WL*40*10) 

    ys1=s1(wavelengths_clip)
    ys2=s2(wavelengths_clip)

    ps=p_map.penalty(s1, s2, wavelengths_clip)
    if(radius_max<4):
        step=1
    else:
        step=radius_max/4    
    step_y, step_x=p_map.step_transform(ps, wavelengths_clip, step) 

#----------Alpha shape maxima selection---------------------

    max_index ,peaks = find_peaks(spectra_clip, height=0, threshold=None, distance=max_vicinity)
    wavelengths_a= np.array(wavelengths_clip)
    max_ys = peaks['peak_heights']
    max_pos = wavelengths_a[max_index]   
    anchors_x, anchors_y, anchors_idx = a_shape.anchors(max_index, spectra_clip, wavelengths_clip, step_y, step_x,min_lambda, radius_min, radius_max, nu, use_pmap, global_stretch)

#------------Outlier removal--------------------------------

    smooth.remove_peaks(anchors_y, anchors_x, anchors_idx, niter_peaks_remove)
    smooth.remove_close(anchors_y, anchors_x)

#--------------Interpolation-------------------------------- 

    if(use_denoise):
        smooth.denoise(anchors_y, anchors_idx, spectra_clip, denoising_distance, max_ys, max_pos, anchors_x)

    if(interp=="cubic"):
        fx=continuum.interpolate(anchors_x, anchors_y, "cubic")
    elif(interp=="linear"):
        fx=continuum.interpolate(anchors_x, anchors_y, "linear")

    y_inter=fx(wavelengths)

#export results to csv

#print("Number of selected maxima:", len(anchors_y))

    np.savetxt(fname="anchors.csv",
          X=np.c_[anchors_x, anchors_y], delimiter=",", header="anchorsX, anchorsY")

#np.savetxt(fname=sys.argv[2],
 #         X=np.c_[wavelengths, y_inter], delimiter=",", header="wave, flux")

#plot
    return wavelengths, spectra, max_pos, max_ys, anchors_x, anchors_y, y_inter, wavelengths_clip, step_x, step_y, ps

#----------------------------------



