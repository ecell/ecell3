import os
p1 = [2.2e-6]
p2 = [0.1, 0.2, 0.3]
k = [2.5e-3]
FileName = ''
for x in p1:
  for y in p2:
    for z in k:
      os.system('ecell3-session --parameters=\"{\'FileName\':\'' + FileName + \
          str(x) + '_' + str(y) + '_' + str(z) + '_visualLog0.dat\',\'p1\':' + \
          str(x) + ',\'p2\':' + str(y) + ',\'k\':' + str(z) +'}\" \
          2012.arjunan.chapter.cluster.py')
