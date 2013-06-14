sim = theSimulator.createStepper('SpatiocyteStepper', 'SS')
sim.VoxelRadius = 4.4e-9 
sim.SearchVacant = 0

theSimulator.rootSystem.StepperID = 'SS'
theSimulator.createEntity('Variable', 'Variable:/:GEOMETRY').Value = 0
theSimulator.createEntity('Variable', 'Variable:/:LENGTHX').Value = 1e-8
theSimulator.createEntity('Variable', 'Variable:/:LENGTHY').Value = 1e-6
theSimulator.createEntity('Variable', 'Variable:/:LENGTHZ').Value = 1e-6
theSimulator.createEntity('Variable', 'Variable:/:VACANT')
theSimulator.createEntity('Variable', 'Variable:/:XYPLANE').Value = 5
theSimulator.createEntity('Variable', 'Variable:/:XZPLANE').Value = 5
theSimulator.createEntity('Variable', 'Variable:/:YZPLANE').Value = 4

# Create the surface compartment:
theSimulator.createEntity('System', 'System:/:Surface').StepperID = 'SS'
theSimulator.createEntity('Variable', 'Variable:/Surface:DIMENSION').Value = 2
theSimulator.createEntity('Variable', 'Variable:/Surface:VACANT')
theSimulator.createEntity('Variable', 'Variable:/Surface:A').Value = 500
theSimulator.createEntity('Variable', 'Variable:/Surface:As').Value = 0

logger = theSimulator.createEntity('VisualizationLogProcess', 'Process:/:logger')
logger.VariableReferenceList = [['_', 'Variable:/Surface:A'], ['_', 'Variable:/Surface:As']]
logger.LogInterval = 0.01


populator = theSimulator.createEntity('MoleculePopulateProcess', 'Process:/:pop')
populator.VariableReferenceList = [['_', 'Variable:/Surface:A']]
populator.VariableReferenceList = [['_', 'Variable:/Surface:As']]

binder = theSimulator.createEntity('DiffusionInfluencedReactionProcess', 'Process:/:reaction1')
binder.VariableReferenceList = [['_', 'Variable:/Surface:A','-1']]
binder.VariableReferenceList = [['_', 'Variable:/Surface:A','-1']]
binder.VariableReferenceList = [['_', 'Variable:/Surface:As','1']]
binder.VariableReferenceList = [['_', 'Variable:/Surface:As','1']]
binder.p = 0.00001

binder = theSimulator.createEntity('DiffusionInfluencedReactionProcess', 'Process:/:reaction2')
binder.VariableReferenceList = [['_', 'Variable:/Surface:A','-1']]
binder.VariableReferenceList = [['_', 'Variable:/Surface:As','-1']]
binder.VariableReferenceList = [['_', 'Variable:/Surface:As','1']]
binder.VariableReferenceList = [['_', 'Variable:/Surface:As','1']]
binder.p = 1

react = theSimulator.createEntity('SpatiocyteNextReactionProcess', 'Process:/:dissocANIOsLip')
react.VariableReferenceList = [['_', 'Variable:/Surface:As', '-1']]
react.VariableReferenceList = [['_', 'Variable:/Surface:A', '1']]
react.Deoligomerize = 6
react.Rates = [243, 81, 27, 9, 3, 1]
#react.k = 100

diffuser = theSimulator.createEntity('DiffusionProcess', 'Process:/:diffuseA')
diffuser.VariableReferenceList = [['_', 'Variable:/Surface:A']]
diffuser.D = 1e-13

run(100)
