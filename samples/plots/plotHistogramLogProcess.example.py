import numpy as np
import os
import pylab as P

labelFontSize = 14
tickFontSize = 14
legendFontSize = 14
lineFontSize = 14

files = []
fileNames = ["../2010.arjunan.syst.synth.biol/HistogramLog.csv"]
legendTitles = []
lines = ['-', '-', '-', '-']
colors = ['r', 'g', 'b', 'k', 'c', 'y']

fig = P.figure(figsize=(10,6))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width*0.7, box.height])
box = ax2.get_position()
ax2.set_position([box.x0, box.y0, box.width*0.7, box.height])
P.xticks(fontsize=tickFontSize)
P.yticks(fontsize=tickFontSize)

for i in range(len(fileNames)):
  f = open(fileNames[i], 'r')
  legendTitles = f.readline().strip().split(",")
  f.close()
  binInterval = float(legendTitles[1])
  data = np.genfromtxt(fileNames[i], delimiter=',', skiprows=1)
  x = []
  currTime = float(data[0][0])
  for row in data:
    if(row[0] == currTime): 
      x.append((float(row[1])+0.5))
    else:
      break
  bins = len(x)
  timePoints = len(data)/bins
  colSize = len(legendTitles)-2
  for l in range(timePoints):
    for m in range(colSize):
      h = []
      y = [0]*bins
      for j in range(int(bins)):
        y[j] = data[l*bins+j][m+2]
        for k in range(int(data[l*bins+j][m+2])):
          h.append(j)
      if(h):
        if(m == 2):
          ax1.hist(h, bins, range=[0,bins], facecolor=colors[i*colSize+m], label=legendTitles[i*colSize+m+2], alpha=0.4)
        ax2.hist(h, bins, range=[0,bins], facecolor=colors[i*colSize+m], label=legendTitles[i*colSize+m+2], alpha=0.4)
      if(m == 2):
        ax1.plot(x, y, ls=lines[0], color=colors[i*colSize+m], label=legendTitles[i*colSize+m+2], linewidth=1)
      ax2.plot(x, y, ls=lines[0], color=colors[i*colSize+m], label=legendTitles[i*colSize+m+2], linewidth=1)
    leg = ax2.legend(bbox_to_anchor=(1.02,2.25), loc='upper left', labelspacing=0.2, handletextpad=0.2, fancybox=True)
    ax1.set_ylim(0)
    ax2.set_ylim(0)
    for t in leg.get_texts():
      t.set_fontsize(legendFontSize)   
    frame = leg.get_frame()
    frame.set_linewidth(None)
    frame.set_facecolor('0.95')
    frame.set_edgecolor('0.75')
    ax2.set_xlabel('Length (x %.2e m)' %(binInterval))
    ax1.set_title('t = %.2e s' %(data[l*bins][0]))
    ax1.set_ylabel('Number of molecules')
    ax2.set_ylabel('Number of molecules')
    ax1.grid(True)
    ax2.grid(True)
    fileName = fileNames[i]+'.%03d.png'%l
    #fileName = fileNames[i]+'.%03d.pdf'%l
    print 'Saving frame', fileName
    fig.savefig(fileName)
    #fig.savefig(fileName, format='pdf')
    ax1.cla()
    ax2.cla()

os.system("ffmpeg -i " + fileNames[0] + ".%03d.png -sameq " + fileNames[0] + ".mp4")



