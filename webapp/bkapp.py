import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, ColumnDataSource, \
                         TapTool, CustomJS, Button, Div, CustomJS, NumericInput, CheckboxGroup, \
                         RadioButtonGroup
from bokeh.plotting import figure
from bokeh.models.widgets import FileInput
from bokeh.events import ButtonClick
import base64
import io
import pandas as pd
from snt_web import *
from os.path import dirname, join
from pathlib import Path

from bisect import bisect_left

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
p.scatter('x', 'y',source=local_max_source ,fill_color = 'yellow', size=8, \
            selection_color="yellow",
            selection_line_color="firebrick",
            selection_line_alpha=1.5,
            selection_line_width=2,
            nonselection_fill_alpha=1,
            nonselection_fill_color="yellow",
            nonselection_line_color="yellow",
            nonselection_line_alpha=1.0)

anchors_data = {'x' : [], 'y': []}
anchors_data_source = ColumnDataSource(data=anchors_data)
p.scatter('x', 'y',source=anchors_data_source,fill_color = 'red', size=8, \
            selection_color="red",
            selection_line_color="black",
            selection_line_alpha=1.5,
            selection_line_width=2.5,
            nonselection_fill_alpha=1,
            nonselection_fill_color="red",
            nonselection_line_color="red",
            nonselection_line_alpha=1.0)                      

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
export_anchors_button.on_click(export_anchors)

def update_radius(attr, old, new):
   global radius_min 
   radius_min = new

def update_radius_max(attr, old, new):
    global radius_max
    radius_max= new

radius_input = NumericInput(low=0.01, title="Radius Min", mode='float',styles={'font-weight': 'bold', 'font-size': '16px'} ,description="α-shape starting radius")
radius_input.on_change("value", update_radius)


radius_max_input= NumericInput(low=0.5, title="Radius Max",mode='float',styles={'font-weight': 'bold', 'font-size': '16px'}, description="α-shape maximum radius" )
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

taptool = p.select(type=TapTool)
#local_max_source.selected.on_change('indices', callbackprint)

def update_anchors():
    selectedIndex = local_max_source.selected.indices
    print(selectedIndex)
    for i  in range (0 ,len(selectedIndex)):
     #   print("Index:", selectedIndex[i])
     #   print("x:", local_max_source.data['x'][selectedIndex[i]])
     #   print("y:", local_max_source.data['y'][selectedIndex[i]])
       # anchors_data_source.data['x'].append(local_max_source.data['x'][selectedIndex[i]])    
       # anchors_data_source.data['y'].append(local_max_source.data['y'][selectedIndex[i]])         
        idx = bisect_left(anchors_data_source.data['x'], local_max_source.data['x'][selectedIndex[i]]) #ordered insertion/removal in anchors list (needed for cubic interp)
        if(idx>=len(anchors_data_source.data['x']) or anchors_data_source.data['x'][idx]!=local_max_source.data['x'][selectedIndex[i]]): #if doesn't exist add it
            anchors_data_source.data['x'].insert(idx, local_max_source.data['x'][selectedIndex[i]])   
            anchors_data_source.data['y'].insert(idx, local_max_source.data['y'][selectedIndex[i]])   
        else:                                                   
            anchors_data_source.data['x'].pop(idx)  #if it exists remove it
            anchors_data_source.data['y'].pop(idx)
    
    local_max_source.selected.indices = []
    anchors_data_source.selected.indices=[]
   # local_max_source.data['x']=local_max_source.data['x']  #dummy code to upload the plots. bokeh? bokeh.
   # local_max_source.data['y']= local_max_source.data['y']                      
    anchors_data_source.data['x']=anchors_data_source.data['x']  #dummy code to update the plots. bokeh? bokeh.
    anchors_data_source.data['y']= anchors_data_source.data['y']

    new_fx=continuum.interpolate(anchors_data_source.data['x'],anchors_data_source.data['y'], interp) #interpolate with new points
    cont_data_source.data['y']=new_fx(cont_data_source.data['x'])               
   

add_maxima_button = Button(label="Add Maxima", button_type="default", width=150)
add_maxima_button.on_event(ButtonClick,update_anchors)

def change_options2(attr, old, new):
    global interp
    if new == 0:
        interp="linear"
    else:
        interp="cubic"    
  
interp_button_title = Div(text="Interpolation",styles={'font-weight': 'bold', 'font-size': '16px'})
LABELS2 = ["Linear", "Cubic"]
interp_button_group = RadioButtonGroup(labels=LABELS2, active=0)
interp_button_group.on_change('active', change_options2)

layout = column(title_text,file_input, row(cont_calc_button, export_button, export_anchors_button), row(p, column( row(radius_input, radius_max_input), checkbox_title, row(checkbox_group, stretching_input),interp_button_title,interp_button_group, add_maxima_button)))

curdoc().add_root(layout)

curdoc().title = "SNT - Spectra Normalization Tool"