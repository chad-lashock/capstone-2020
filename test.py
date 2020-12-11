# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:07:11 2020

@author: 80lascha
"""

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
import pandas as pd

fig = plt.figure(figsize=(12,6))
ax = plt.axes()
ax.set_facecolor('whitesmoke')



lines = [z[0:6],z[6:10]]

line_segments = LineCollection(lines, colors=[mcolors.to_rgb("blue"),
                                              mcolors.to_rgb("green")])

ax.add_collection(line_segments)
ax.autoscale()

plt.grid()
plt.xlabel("Date")
plt.ylabel("Price ($)")

plt.show()