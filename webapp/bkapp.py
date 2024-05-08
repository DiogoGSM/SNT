import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, ColumnDataSource, \
                         TapTool, CustomJS, Button, Div, CustomJS, NumericInput, CheckboxGroup
from bokeh.plotting import figure
from bokeh.models.widgets import FileInput
import base64
import io
import pandas as pd
from snt_web import *
from os.path import dirname, join
from pathlib import Path

remove_n_first=0      #removes the first n points in the spectra (only if first points are not useful!) otherwise set to 0
radius_min=20             #min alpha shape radius
radius_max=40            #max alpha shape radius (should be at least the size of the largest gap)
max_vicinity = 10       #required number of values between adjacent maxima
stretching = 10      #normalization parameter
use_RIC = False  #use RIC to avoid large "dips" in the continuum (see documentation)
interp="linear"     #interpolation type
use_denoise=False  #for noisy spectra use the average of the flux value around the maximum
usefilter= False #use savgol filter to smooth spectra 
nu=1                    #exponent of the computed penalty (see documentation)                  
niter_peaks_remove=0 #number of iterations to remove sharpest peaks before interpolation
denoising_distance=5

tool_list=["tap","pan","zoom_in","zoom_out","box_zoom","reset"]

title_text = Div(text="<h1>SNT - Spectra Normalization Tool</h1>")

data = {'x': [], 'y': []}
source = ColumnDataSource(data=data)
p = figure(width=800, height=600,
             tools=tool_list)
p.line('x', 'y', source=source)
p.xaxis.axis_label = 'Wavelengths (Å)'
p.yaxis.axis_label = 'Flux'
p.xaxis.axis_label_text_font_size = '13pt'  
p.yaxis.axis_label_text_font_size = '13pt'
p.xaxis.axis_label_text_font_style = 'normal'
p.yaxis.axis_label_text_font_style = 'normal'
p.xaxis.major_label_text_font_size = '11pt'  
p.yaxis.major_label_text_font_size = '11pt' 


local_max_data = {'x' : [], 'y': []}
local_max_source = ColumnDataSource(data=local_max_data)
p.scatter('x', 'y',source=local_max_source ,fill_color = 'yellow', size=6)

anchors_data = {'x' : [], 'y': []}
anchors_data_source = ColumnDataSource(data=anchors_data)
p.scatter('x', 'y',source=anchors_data_source,fill_color = 'red', size=6)

cont_data = {'x': [], 'y': []}
cont_data_source = ColumnDataSource(data=cont_data)
p.line('x', 'y', source= cont_data_source, line_color = 'red')

def upload_plot_data(attr, old, new):
    anchors_data_source.data= {'x' : [], 'y': []}   #reset sources on new upload
    local_max_source.data= {'x' : [], 'y': []}
    cont_data_source.data = {'x': [], 'y': []}
    print("in upload callback")
    decoded = base64.b64decode(new)
    data = io.BytesIO(decoded)
    df = pd.read_csv(data)
    print("shape=", df.shape)
    if(df.shape[1]==3):                    #if it includes indexes in the csv drop them
        df.drop(columns=df.columns[0], axis=1, inplace=True)      
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]     
    source.data= {'x': x, 'y': y}

file_input = FileInput(accept=".csv", width=300, height=50)
file_input.on_change('value', upload_plot_data)


def calc_func():
    wavelengths, spectra, max_pos, max_ys, anchors_x, anchors_y, y_inter, wavelengths_clip, step_x, step_y, ps = \
            cont_determ (source.data['x'].tolist(), source.data['y'].tolist(), remove_n_first, radius_min, radius_max, max_vicinity, \
                        stretching, use_RIC, interp, use_denoise, usefilter, nu, niter_peaks_remove, denoising_distance)
    local_max_source.data = {'x': max_pos, 'y': max_ys}
    anchors_data_source.data= {'x': anchors_x, 'y': anchors_y}
    cont_data_source.data = {'x': wavelengths, 'y': y_inter}

def export_line():
    pass

def export_anchors():
    pass

cont_calc_button = Button(label="Display Continuum", button_type="primary", width=150)
cont_calc_button.on_click(calc_func)

export_button = Button(label="Export line", button_type="primary", width=150)
export_button.js_on_event(
    "button_click",
    CustomJS(
        args=dict(source=cont_data_source),
        code=(Path(__file__).parent / "download.js").read_text("utf8"),
    ),
)



export_anchors_button= Button(label="Export Anchors", button_type="primary", width=150)
export_anchors_button.on_click(export_line)

def update_radius(attr, old, new):
   global radius_min 
   radius_min = new

def update_radius_max(attr, old, new):
    global radius_max
    radius_max= new

radius_input = NumericInput(low=1, title="Radius Min", mode='float',styles={'font-weight': 'bold', 'font-size': '16px'} ,description="α-shape starting radius")
radius_input.on_change("value", update_radius)


radius_max_input= NumericInput(low=1, title="Radius Max",mode='float',styles={'font-weight': 'bold', 'font-size': '16px'}, description="α-shape maximum radius" )
radius_max_input.on_change("value", update_radius_max)

def change_options(attr, old, new):
    global use_denoise
    global use_RIC  
    global usefilter
    if 0 in new:
        use_denoise=True
    else:
        use_denoise=False
    if 1 in new:
        usefilter=True
    else:
        usefilter=False
    if 2 in new:
        use_RIC=True
    else:
        use_RIC=False

LABELS = ["Denoise", "Savitzky-golay filter", "RIC"]

checkbox_group = CheckboxGroup(labels=LABELS, styles={'font-size': '15px'})
checkbox_group.on_change('active', change_options)

checkbox_title = Div(text="Options",styles={'font-weight': 'bold', 'font-size': '16px'})

def update_stretching(attr, old, new):
    global stretching
    stretching=new

stretching_input= NumericInput(low=0, mode='float',title="Stretching",styles={'font-weight': 'bold', 'font-size': '16px'}, description="Vertical axis scaling" )
stretching_input.on_change("value", update_stretching)


#source = ColumnDataSource(data=dict(x=[1, 2, 3, 4, 5], y=[2, 5, 8, 2, 7]))

#p.scatter(x='x', y='y', size=20, source=source)


#def callbackprint(attrname, old, new):
#    selectedIndex = source.selected.indices
#    for i  in range (0 ,len(selectedIndex)):
#        print("Index:", selectedIndex[i])
#        print("x:", source.data['x'][selectedIndex[i]])
#        print("y:", source.data['y'][selectedIndex[i]])
#    print("-------------------------------------")
#taptool = p.select(type=TapTool)
#source.selected.on_change('indices', callbackprint)

layout = column(title_text,file_input, row(cont_calc_button, export_button, export_anchors_button), row(p, column( row(radius_input, radius_max_input), checkbox_title, row(checkbox_group, stretching_input))))

curdoc().add_root(layout)

curdoc().title = "SNT - Spectra Normalization Tool"