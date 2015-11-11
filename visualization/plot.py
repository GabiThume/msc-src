from __future__ import division

import numpy as np
import pandas as pd
import csv
from bokeh.plotting import output_file, figure
from sklearn.preprocessing import LabelEncoder
from sklearn.lda import LDA

H = 10
W = 10

p = figure(x_range=[0,10], y_range=[0,10])

url = ["../test/Lenna.png"]
# p.image_url(x=list(range(0,100,10)), y=list(range(0,100,10)), url=url, global_alpha=0.2, h = 10, w = 10)
p.image_url(x=0, y=0, url=url, h=H, w=W)

# Open in a browser
show(p)

# features = []
# csvfile = open("visualization/images/features/BIC_Intensity_64c_100r_200i_artificial.csv", "rb")
# reader = csv.reader(csvfile, delimiter= ' ')
# for row in reader:
#     features.append(row)

df = pd.io.parsers.read_csv(filepath_or_buffer='visualization/images/features/BIC_Intensity_64c_100r_200i_artificial.csv', header=None, sep = ' ')
df.columns = ['Image label'] + ['Class label'] + ['Train or test'] + [l for l in range(0,128)]
X = df[range(3, 127)].values
y = df['Class label'].values

 sklearn_lda = LDA(n_components=2)
 X_lda = sklearn_lda.fit_transform(X,y)

 #dropna
