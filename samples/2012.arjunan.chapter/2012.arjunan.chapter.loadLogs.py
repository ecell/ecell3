import glob
import os
files = glob.glob('*0.dat')
for i in files:
  print "\nloading file " + i + "..." 
  os.system('./spatiocyte ' + i)
