import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, ColumnDataSource, TapTool, CustomJS
from bokeh.plotting import figure


tool_list=["tap","pan","zoom_in","zoom_out","box_zoom","reset"]

p = figure(width=800, height=600,
             tools=tool_list)

source = ColumnDataSource(data=dict(x=[1, 2, 3, 4, 5], y=[2, 5, 8, 2, 7]))

p.scatter(x='x', y='y', size=20, source=source)


def callbackprint(attrname, old, new):
    selectedIndex = source.selected.indices
    for i  in range (0 ,len(selectedIndex)):
        print("Index:", selectedIndex[i])
        print("x:", source.data['x'][selectedIndex[i]])
        print("y:", source.data['y'][selectedIndex[i]])
    print("-------------------------------------")
taptool = p.select(type=TapTool)
source.selected.on_change('indices', callbackprint)

curdoc().add_root(p)
curdoc().title = "servertest"