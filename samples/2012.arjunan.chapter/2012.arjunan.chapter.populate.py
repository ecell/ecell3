# Example of python scripting to populate molecules at the poles of a rod compartment
theSimulator.createStepper('SpatiocyteStepper', 'SS').VoxelRadius = 8e-8
# Create the root container compartment using the rod geometry:
theSimulator.rootSystem.StepperID = 'SS'
theSimulator.createEntity('Variable', 'Variable:/:GEOMETRY').Value = 3
theSimulator.createEntity('Variable', 'Variable:/:LENGTHX').Value = 10e-6
theSimulator.createEntity('Variable', 'Variable:/:LENGTHY').Value = 2e-6
theSimulator.createEntity('Variable', 'Variable:/:VACANT')
logger = theSimulator.createEntity('VisualizationLogProcess', 'Process:/:logger')
logger.LogInterval = 1
logger.VariableReferenceList = [['_', 'Variable:/Surface:A'], ['_', 'Variable:/Surface:B']]
populator = theSimulator.createEntity('MoleculePopulateProcess', 'Process:/:populateLeft')
populator.VariableReferenceList = [['_', 'Variable:/Surface:A']]
populator.OriginX = -1
populator.UniformRadiusX = 0.5
populator = theSimulator.createEntity('MoleculePopulateProcess', 'Process:/:populateRight')
populator.VariableReferenceList = [['_', 'Variable:/Surface:B']]
populator.OriginX = 1
populator.UniformRadiusX = 0.5
# Create the surface compartment:
theSimulator.createEntity('System', 'System:/:Surface').StepperID = 'SS'
theSimulator.createEntity('Variable', 'Variable:/Surface:DIMENSION').Value = 2
theSimulator.createEntity('Variable', 'Variable:/Surface:VACANT')
theSimulator.createEntity('Variable', 'Variable:/Surface:A').Value = 500
theSimulator.createEntity('Variable', 'Variable:/Surface:B').Value = 500
run(100)
