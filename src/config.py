#---------------------------------
#Parameters
#---------------------------------

remove_n_first=0      #removes the first n points in the spectra (only if first points are not useful!) otherwise set to 0

radius_min=1             #min alpha shape radius
radius_max=2            #max alpha shape radius (should be at least the size of the largest gap)
max_vicinity = 1       #required number of values between adjacent maxima
stretching = 10      #normalization parameter
use_RIC = True  #use RIC to avoid large "dips" in the continuum (see documentation)
interp="linear"     #interpolation type
use_denoise=True  #for noisy spectra use the average of the flux value around the maximum
usefilter= True #use savgol filter to smooth spectra 
nu=1                    #exponent of the computed penalty (see documentation)                  
niter_peaks_remove=0 #number of iterations to remove sharpest peaks before interpolation
denoising_distance=5   #number of points to calculate the average around a maximum if use_denoise is True, useful for noisy spectra
