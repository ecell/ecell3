import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import pylab as P

labelFontSize = 14
tickFontSize = 14
legendFontSize = 14
lineFontSize = 14

files = []
fileNames = ["../2010.arjunan.syst.synth.biol/CoordinateLog.csv"]
legendTitles = []
lines = ['-', '-', '-', '-']
colors = ['r', 'g', 'b', 'y', 'c', 'k']

fig = P.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
P.xticks(fontsize=tickFontSize)
P.yticks(fontsize=tickFontSize)

for i in range(len(fileNames)):
  f = open(fileNames[i], 'r')
  legendTitles = f.readline().strip().split(",")
  logInterval = float(legendTitles[0].split("=")[1])
  worldWidth = float(legendTitles[1].split("=")[1])
  worldHeight = float(legendTitles[2].split("=")[1])
  worldLength = float(legendTitles[3].split("=")[1])
  voxelRadius = float(legendTitles[4].split("=")[1])
  speciesNames = []
  speciesRadii = []
  for j in range(len(legendTitles)-5):
    speciesNames.append(legendTitles[j+5].split("=")[0])
    speciesRadii.append(float(legendTitles[j+5].split("=")[1]))
  speciesSize = len(speciesNames)
  logCnt = 0
  lineCnt = 0
  markers = []
  for line in f:
    coords = line.strip().split(",")
    time = float(coords[0])
    x = []
    y = []
    z = []
    for l in range((len(coords)-1)/3):
      x.append(float(coords[l*3+1]))
      y.append(float(coords[l*3+2]))
      z.append(float(coords[l*3+3]))
    ax.scatter(x, z, y, color=colors[lineCnt])
    markers.append(P.Rectangle((0, 0), 1, 1, fc=colors[lineCnt]))
    lineCnt = lineCnt + 1
    if lineCnt == speciesSize:
      ax.set_xlabel('Length (x %.2e m)' %(voxelRadius*2))
      ax.set_ylabel('Width (x %.2e m)' %(voxelRadius*2))
      ax.set_zlabel('Height (x %.2e m)' %(voxelRadius*2))
      ax.set_title('t = %.2e s' %time)
      leg = ax.legend(markers, speciesNames, bbox_to_anchor=(1.0,0.95), loc='upper left', labelspacing=0.2, handletextpad=0.2, fancybox=True)
      for t in leg.get_texts():
        t.set_fontsize(legendFontSize)   
      frame = leg.get_frame()
      frame.set_linewidth(None)
      frame.set_facecolor('0.95')
      frame.set_edgecolor('0.75')
      fileName = fileNames[i]+'.%03d.png'%logCnt
      print 'Saving frame', fileName
      fig.savefig(fileName)
      markers = []
      ax.cla()
      logCnt = logCnt + 1
      lineCnt = 0

os.system("ffmpeg -i " + fileNames[0] + ".%03d.png -sameq " + fileNames[0] + ".mp4")



