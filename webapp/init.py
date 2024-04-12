from snt import *
import matplotlib
import io
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpld3

import numpy as np
from flask import Flask, render_template, request

app=Flask(__name__)

 
remove_n_first=0      #removes the first n points in the spectra (only if first points are not useful!) otherwise set to 0
#radius_min=1             #min alpha shape radius
#radius_max=2            #max alpha shape radius (should be at least the size of the largest gap)
#max_vicinity = 1       #required number of values between adjacent maxima
#stretching = 10      #normalization parameter
use_RIC = True  #use RIC to avoid large "dips" in the continuum (see documentation)
interp="linear"     #interpolation type
use_denoise=True  #for noisy spectra use the average of the flux value around the maximum
usefilter= True #use savgol filter to smooth spectra 
nu=1                    #exponent of the computed penalty (see documentation)                  
niter_peaks_remove=0 #number of iterations to remove sharpest peaks before interpolation
denoising_distance=5

def on_select(brush):
    ind = brush.selected
    selected_data = {'x': [], 'y': []}
    for i in ind:
        selected_data['x'].append(x[i])
        selected_data['y'].append(y[i])
    print("Selected data:", selected_data)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_parameters", methods=['GET','POST'])
def get_parameters():
    if request.method == "POST":
        radius_min=  request.form.get('radius_min', type=int)
        radius_max=  request.form.get('radius_max', type=int)
        max_vicinity = request.form.get('max_vicinity', type=int)
        stretching = request.form.get('stretching', type=int)

    wavelengths, spectra, max_pos, max_ys, anchors_x, anchors_y, y_inter, wavelengths_clip, step_x, step_y, ps = \
    cont_determ (remove_n_first, radius_min, radius_max, max_vicinity, \
                    stretching, use_RIC, interp, use_denoise, usefilter, nu, niter_peaks_remove, denoising_distance)
    
   
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(wavelengths, spectra)                     
    ax.scatter(max_pos, max_ys, color='g', s=5)
    ax.scatter(anchors_x, anchors_y, color ='r', s=5)
    ax.plot(wavelengths, y_inter, color='r')
    ax.set_ylabel("flux")
    ax.set_xlabel(r"wavelengths($\AA$)", fontsize=17)
   # ax[1].plot(wavelengths_clip, ps)
   # ax[1].plot(step_x, step_y, color='black')
   # ax[1].set_xlabel("wavelengths")
   # ax[1].set_ylabel("RIC")

    html_fig = mpld3.fig_to_html(fig)
    plt.close(fig)
    return render_template('index.html', plot=html_fig)




if __name__ == "__main__":
    app.run()