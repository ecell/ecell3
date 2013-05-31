# Example of python scripting to create a neuron with 5 minor processes
theSimulator.createStepper('SpatiocyteStepper', 'SS').VoxelRadius = 10e-8
# Create the root container compartment using the default Cuboid geometry:
theSimulator.rootSystem.StepperID = 'SS'
theSimulator.createEntity('Variable', 'Variable:/:LENGTHX').Value = 61e-6
theSimulator.createEntity('Variable', 'Variable:/:LENGTHY').Value = 25e-6
theSimulator.createEntity('Variable', 'Variable:/:LENGTHZ').Value = 5.5e-6
theSimulator.createEntity('Variable', 'Variable:/:VACANT')
logger = theSimulator.createEntity('VisualizationLogProcess', 'Process:/:logger')
logger.LogInterval = 1
logger.VariableReferenceList = [['_', 'Variable:/Soma/Membrane:VACANT'], ['_', 'Variable:/Soma:K']]
logger.VariableReferenceList = [['_', 'Variable:/Dendrite%d/Membrane:VACANT' %i] for i in range(5)] 
populator = theSimulator.createEntity('MoleculePopulateProcess', 'Process:/:populate')
populator.VariableReferenceList = [['_', 'Variable:/Soma:K']]
# Create the Soma compartment of the Neuron:
theSimulator.createEntity('System', 'System:/:Soma').StepperID = 'SS'
theSimulator.createEntity('Variable', 'Variable:/Soma:GEOMETRY').Value = 1
theSimulator.createEntity('Variable', 'Variable:/Soma:LENGTHX').Value = 10e-6
theSimulator.createEntity('Variable', 'Variable:/Soma:LENGTHY').Value = 10e-6
theSimulator.createEntity('Variable', 'Variable:/Soma:LENGTHZ').Value = 6.5e-6
theSimulator.createEntity('Variable', 'Variable:/Soma:ORIGINX').Value = -0.48 
theSimulator.createEntity('Variable', 'Variable:/Soma:ORIGINY').Value = -0.2
theSimulator.createEntity('Variable', 'Variable:/Soma:ORIGINZ').Value = -0.6
theSimulator.createEntity('Variable', 'Variable:/Soma:VACANT')
theSimulator.createEntity('Variable', 'Variable:/Soma:K').Value = 1000
diffuser = theSimulator.createEntity('DiffusionProcess', 'Process:/Soma:diffuseK')
diffuser.VariableReferenceList = [['_', 'Variable:.:K']]
diffuser.D = 0.2e-12
# Create the Soma membrane:
theSimulator.createEntity('System', 'System:/Soma:Membrane').StepperID = 'SS'
theSimulator.createEntity('Variable', 'Variable:/Soma/Membrane:DIMENSION').Value = 2
theSimulator.createEntity('Variable', 'Variable:/Soma/Membrane:VACANT')
# Parameters of Dendrites/Minor Processes:
dendritesLengthX = [40e-6, 10e-6, 10e-6, 10e-6, 10e-6]
dendritesOriginX = [0.32, -0.78, -0.48, -0.3, -0.66] 
dendritesOriginY = [-0.2, -0.2, 0.52, -0.65, -0.65]
dendritesRotateZ = [0, 0, 1.57, 0.78, -0.78]
for i in range(5):
  # Create the Dendrite:
  theSimulator.createEntity('System', 'System:/:Dendrite%d' %i).StepperID = 'SS'
  theSimulator.createEntity('Variable', 'Variable:/Dendrite%d:GEOMETRY' %i).Value = 3 
  theSimulator.createEntity('Variable', 'Variable:/Dendrite%d:LENGTHX' %i).Value = dendritesLengthX[i]
  theSimulator.createEntity('Variable', 'Variable:/Dendrite%d:LENGTHY' %i).Value = 1.5e-6
  theSimulator.createEntity('Variable', 'Variable:/Dendrite%d:ORIGINX' %i).Value = dendritesOriginX[i]
  theSimulator.createEntity('Variable', 'Variable:/Dendrite%d:ORIGINY' %i).Value = dendritesOriginY[i]
  theSimulator.createEntity('Variable', 'Variable:/Dendrite%d:ORIGINZ' %i).Value = -0.6
  theSimulator.createEntity('Variable', 'Variable:/Dendrite%d:ROTATEZ' %i).Value = dendritesRotateZ[i]
  theSimulator.createEntity('Variable', 'Variable:/Dendrite%d:VACANT' %i)
  theSimulator.createEntity('Variable', 'Variable:/Dendrite%d:DIFFUSIVE' %i).Name = '/:Soma'
  # Create the Dendrite membrane:
  theSimulator.createEntity('System', 'System:/Dendrite%d:Membrane' %i).StepperID = 'SS'
  theSimulator.createEntity('Variable', 'Variable:/Dendrite%d/Membrane:DIMENSION' %i).Value = 2
  theSimulator.createEntity('Variable', 'Variable:/Dendrite%d/Membrane:VACANT' %i)
  theSimulator.createEntity('Variable', 'Variable:/Dendrite%d/Membrane:DIFFUSIVE' %i).Name = '/Soma:Membrane'
run(100)
