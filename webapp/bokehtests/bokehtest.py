import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, ColumnDataSource, \
                         TapTool, CustomJS, Button, Div
from bokeh.plotting import figure
from bokeh.models.widgets import FileInput
import base64
import io
import pandas as pd

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

def upload_plot_data(attr, old, new):
    decoded = base64.b64decode(new)
    data = io.BytesIO(decoded)
    df = pd.read_csv(data)
    if(df.shape[1]==3):                    #if it includes indexes in the csv drop them
        df.drop(columns=df.columns[0], axis=1, inplace=True)      
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]   
    source.data= {'x': x, 'y': y}

file_input = FileInput(accept=".csv", width=300, height=50)
file_input.on_change('value', upload_plot_data)


def calc_func():
    pass

def export_line():
    pass

def export_anchors():
    pass

cont_calc_button = Button(label="Display Continuum", button_type="primary", width=150)
cont_calc_button.on_click(calc_func)

export_button = Button(label="Export line", button_type="primary", width=150)
export_button.on_click(export_line)

export_anchors_button= Button(label="Export Anchors", button_type="primary", width=150)
export_anchors_button.on_click(export_line)


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

layout = column(title_text,file_input, row(cont_calc_button, export_button, export_anchors_button), p)

curdoc().add_root(layout)

curdoc().title = "SNT - Spectra Normalization Tool"