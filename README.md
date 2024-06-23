# SNT
SNT - Spectra Normalization Tool

### Requirements  
 
* Astropy (https://www.astropy.org/)
* Matplotlib (https://matplotlib.org/)
* Numpy (https://numpy.org/)
* Scipy (https://scipy.org/)
* Pandas (https://pandas.pydata.org/)
* Bokeh ([https://bokeh.org](https://docs.bokeh.org/en/latest/docs/first_steps.html)) - only for the webapp interface

### How to run

### Webapp interface

Go to SNT/webapp folder and run in terminal:
```console
chmod 755 run_server.sh
./run_server.sh
```
The first command only needs to be run when running the program for the first time.

(Windows users - just copy the command in .sh file to the terminal)

### Command line

Go to SNT/cmdversion and run in terminal:

```console
python snt.py spec.csv
```
The results will be in continuum.csv and the selected anchors in anchors.csv.

Parameters for this version should be changed in the "config.py" file.

