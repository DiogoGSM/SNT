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
from pandas import *
 
inputfile=sys.argv[1]

df = read_csv(inputfile)
 
if(df.shape[1]==3):                    #if it includes indexes in the csv drop them
        df.drop(columns=df.columns[0], axis=1, inplace=True)      
x = df.iloc[:, 0]
y = df.iloc[:, 1]  

# converting column data to list
wavelengths = x.tolist()
spectra = y.tolist()


#with fits.open(inputfile) as hdu:

#    full_data = hdu[1].data
#    header = hdu[0].header

#wavelengths = full_data["wavelength"]
#spectra = full_data["flux"]
#uncertainties = full_data["error"]   

#np.savetxt(fname="tau-ceti_big.csv",
#          X=np.c_[wavelengths, spectra], delimiter=",", header="wave, flux")


min_lambda = min(wavelengths)
#FWHM=header['HIERARCH ESO QC CCF FWHM']   #FWHM in Km/s                            
#FWHM_WL= min_lambda*(FWHM/(constant.c/1000)) #FWHM in A
FWHM_WL=0.1 #in case the spectre doesn't include information about FWHM

#plt.plot(wavelengths, spectra)
#plt.show()
#---------------------------------
#Parameters (change values in config.py)
#---------------------------------
remove_n_first=config.remove_n_first      #removes the first n points in the spectra (only if first points are not useful!) otherwise set to 0
radius_min=config.radius_min              #min alpha shape radius
radius_max=config.radius_max           #max alpha shape radius (should be at least the size of the largest gap)
max_vicinity = config.max_vicinity        #required number of values between adjacent maxima
global_stretch =config.stretching      #normalization parameter
use_pmap = config.use_RIC  #use the RIC to avoid large "dips" in the continuum (see documentation)
interp=config.interp    #interpolation type
use_denoise=config.use_denoise  #for noisy spectra use the average of the flux value around the maximum
usefilter=config.usefilter   #use savgol filter to smooth spectra 
nu=config.nu                    #exponent of the computed penalty (see documentation)                  
niter_peaks_remove=config.niter_peaks_remove  #number of iterations to remove sharpest peaks before interpolation
denoising_distance=config.denoising_distance   #number of points to calculate the average around a maximum if use_denoise is True, useful for noisy spectra

#---------------------------------
print("Running...")
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
    smooth.denoise(anchors_y, anchors_idx, spectra_clip, denoising_distance)

if(interp=="cubic"):
    fx=continuum.interpolate(anchors_x, anchors_y, "cubic")
elif(interp=="linear"):
    fx=continuum.interpolate(anchors_x, anchors_y, "linear")

y_inter=fx(wavelengths)

#export results to csv

print("Number of selected maxima:", len(anchors_y))

np.savetxt(fname="anchors.csv",
          X=np.c_[anchors_x, anchors_y], delimiter=",", header="anchorsX, anchorsY")

np.savetxt(fname="continuum.csv",
          X=np.c_[wavelengths, y_inter], delimiter=",", header="wave, flux")

#plot
fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(wavelengths, spectra)                     
ax[0].scatter(max_pos, max_ys, color='g', s=15)
ax[0].scatter(anchors_x, anchors_y, color ='r', s=20)
ax[0].plot(wavelengths, y_inter, color='r')
ax[0].set_ylabel("flux")
ax[1].plot(wavelengths_clip, ps)
ax[1].plot(step_x, step_y, color='black')
ax[1].set_xlabel("wavelengths")
ax[1].set_ylabel("RIC")
plt.show()   
#----------------------------------



