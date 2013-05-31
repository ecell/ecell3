import numpy
import csv
import math
from matplotlib import rc
from pylab import *
from matplotlib.ticker import MaxNLocator

labelFontSize = 14
tickFontSize = 14
legendFontSize = 14
lineFontSize = 14

fileNames = ['IterateLog.csv']
legendTitles = ['State 2', 'State 1']
speciesList = ['E']
lines = ['-', '--', '-', '-']
colors = ['r', 'b', '#000000', 'black', 'c', 'k', '#009955', '#ff9933', '#ff00ff', '#11dd00']
data = genfromtxt(fileNames[0], delimiter=',').T

colSize = len(data)-1
for i in range(colSize):
  plot(data[0], data[i+1], ls=lines[0], color=colors[i], label=legendTitles[0], linewidth=1)

ax = gca()
ax.grid(color='b', linestyle='--')
#ax.yaxis.set_major_locator(MaxNLocator(14))
leg = legend(loc=0, labelspacing=0.2, handletextpad=0.2, fancybox=True)
for t in leg.get_texts():
  t.set_fontsize(legendFontSize)   
xticks(fontsize=tickFontSize)
yticks(fontsize=tickFontSize)
frame = leg.get_frame()
frame.set_linewidth(None)
frame.set_facecolor('0.95')
frame.set_edgecolor('0.75')
xlabel('site')
ylabel('density')
show()

