# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:03:43 2022

@author: Wladek
"""

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
import numpy as np

df = pd.read_excel('channels_list.xlsx')

df.index = [item for item in df.index] # Format date

fig, ax = plt.subplots(figsize=(12, 2)) # set size frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
tabla = table(ax, df, loc='upper right', colWidths=[0.17]*len(df.columns))  # where df is your data frame
tabla.auto_set_font_size(False) # Activate set fontsize manually
tabla.set_fontsize(12) # if ++fontsize is necessary ++colWidths
tabla.scale(1.2, 1.2) # change size table
plt.savefig('table.png', transparent=True)