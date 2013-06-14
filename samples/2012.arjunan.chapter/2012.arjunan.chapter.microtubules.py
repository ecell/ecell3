import math
import random
minDist = 75e-9
dendriteRadius = 0.75e-6
dendriteLength = 10e-6
lengths = [8.4e-6, 6.3e-6, 4.2e-6, 2.1e-6, 1e-6]
lengthFreqs = [7, 10, 11, 21, 108]
mtOriginX = []
mtOriginZ = []
mtOriginY = []
expandedLengths = []

def isSpacedOut(x, y, z, length):
  for i in range(len(expandedLengths)-1):
    maxOriX = mtOriginX[i]*dendriteLength/2 + expandedLengths[i]/2
    minOriX = mtOriginX[i]*dendriteLength/2 - expandedLengths[i]/2
    maxX = x*dendriteLength/2 + length/2
    minX = x*dendriteLength/2 - length/2
    y2 = math.pow((y-mtOriginY[i])*dendriteRadius, 2)
    z2 = math.pow((z-mtOriginZ[i])*dendriteRadius, 2)
    if((minX <= maxOriX or maxX >= minOriX) and math.sqrt(y2+z2) < minDist):
      return False
    elif(minX > maxOriX and math.sqrt(y2+z2+math.pow(minX-maxOriX, 2)) < minDist):
      return False
    elif(maxX < minOriX and math.sqrt(y2+z2+math.pow(maxX-minOriX, 2)) < minDist):
      return False
  return True

for i in range(len(lengthFreqs)):
  maxX = (dendriteLength-lengths[i])/dendriteLength
  for j in range(int(lengthFreqs[i])):
    expandedLengths.append(lengths[i])
    x = random.uniform(-maxX, maxX)
    y = random.uniform(-0.95, 0.95)
    z = random.uniform(-0.95, 0.95)
    while(y*y+z*z > 0.9 or not isSpacedOut(x, y, z, lengths[i])):
      x = random.uniform(-maxX, maxX)
      y = random.uniform(-0.95, 0.95)
      z = random.uniform(-0.95, 0.95)
    mtOriginX.append(x)
    mtOriginY.append(y)
    mtOriginZ.append(z)

theSimulator.createStepper('SpatiocyteStepper', 'SS').VoxelRadius = 0.8e-8
theSimulator.rootSystem.StepperID = 'SS'
theSimulator.createEntity('Variable', 'Variable:/:GEOMETRY').Value = 3
theSimulator.createEntity('Variable', 'Variable:/:LENGTHX').Value = dendriteLength
theSimulator.createEntity('Variable', 'Variable:/:LENGTHY').Value = dendriteRadius*2
theSimulator.createEntity('Variable', 'Variable:/:VACANT')
theSimulator.createEntity('Variable', 'Variable:/:K').Value = 100
diffuser = theSimulator.createEntity('DiffusionProcess', 'Process:/:diffuseK')
diffuser.VariableReferenceList = [['_', 'Variable:/:K']]
diffuser.D = 0.2e-12
visualLogger = theSimulator.createEntity('VisualizationLogProcess', 'Process:/:visualLogger')
visualLogger.LogInterval = 1
visualLogger.VariableReferenceList = [['_', 'Variable:/Membrane:VACANT'], ['_', 'Variable:/:K']]
theSimulator.createEntity('MoleculePopulateProcess', 'Process:/:populate').VariableReferenceList = [['_', 'Variable:/:K']]
theSimulator.createEntity('System', 'System:/:Membrane').StepperID = 'SS'
theSimulator.createEntity('Variable', 'Variable:/Membrane:DIMENSION').Value = 2
theSimulator.createEntity('Variable', 'Variable:/Membrane:VACANT')
for i in range(len(expandedLengths)):
  theSimulator.createEntity('System', 'System:/:Microtubule%d' %i).StepperID = 'SS'
  theSimulator.createEntity('Variable', 'Variable:/Microtubule%d:GEOMETRY' %i).Value = 2
  theSimulator.createEntity('Variable', 'Variable:/Microtubule%d:LENGTHX' %i).Value = expandedLengths[i]
  theSimulator.createEntity('Variable', 'Variable:/Microtubule%d:LENGTHY' %i).Value = 6e-9
  theSimulator.createEntity('Variable', 'Variable:/Microtubule%d:ORIGINX' %i).Value = mtOriginX[i]
  theSimulator.createEntity('Variable', 'Variable:/Microtubule%d:ORIGINY' %i).Value = mtOriginY[i]
  theSimulator.createEntity('Variable', 'Variable:/Microtubule%d:ORIGINZ' %i).Value = mtOriginZ[i]
  theSimulator.createEntity('Variable', 'Variable:/Microtubule%d:VACANT' %i)
  theSimulator.createEntity('System', 'System:/Microtubule%d:Membrane' %i).StepperID = 'SS'
  theSimulator.createEntity('Variable', 'Variable:/Microtubule%d/Membrane:DIMENSION' %i).Value = 2
  theSimulator.createEntity('Variable', 'Variable:/Microtubule%d/Membrane:VACANT' %i)
  visualLogger.VariableReferenceList = [['_', 'Variable:/Microtubule%d/Membrane:VACANT' %i]] 
run(100)
