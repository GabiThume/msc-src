from __future__ import division

import numpy as np
from bokeh.plotting import output_file, figure

H = 10
W = 10

p = figure(x_range=[0,10], y_range=[0,10])

url = ["../test/Lenna.png"]
# p.image_url(x=list(range(0,100,10)), y=list(range(0,100,10)), url=url, global_alpha=0.2, h = 10, w = 10)
p.image_url(x=0, y=0, url=url, h=H, w=W)

# Open in a browser
show(p)
