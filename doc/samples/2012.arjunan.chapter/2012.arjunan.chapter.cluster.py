message('\nrunning: ' + FileName)
theSimulator.createStepper('SpatiocyteStepper', 'SS').VoxelRadius = 0.5
# Create the system compartment:
theSimulator.rootSystem.StepperID = 'SS'
theSimulator.createEntity('Variable', 'Variable:/:LENGTHX').Value = 250
theSimulator.createEntity('Variable', 'Variable:/:LENGTHY').Value = 250
theSimulator.createEntity('Variable', 'Variable:/:LENGTHZ').Value = 20
theSimulator.createEntity('Variable', 'Variable:/:VACANT')
theSimulator.createEntity('Variable', 'Variable:/:XYPLANE').Value = 3
theSimulator.createEntity('Variable', 'Variable:/:YZPLANE').Value = 5
theSimulator.createEntity('Variable', 'Variable:/:XZPLANE').Value = 5
logger = theSimulator.createEntity('VisualizationLogProcess', 'Process:/:logger')
logger.LogInterval = 500
logger.VariableReferenceList = [['_', 'Variable:/Surface:A'], ['_', 'Variable:/Surface:Ac']]
logger.FileName = FileName
# Create the surface compartment:
theSimulator.createEntity('System', 'System:/:Surface').StepperID = 'SS'
theSimulator.createEntity('Variable', 'Variable:/Surface:DIMENSION').Value = 2
theSimulator.createEntity('Variable', 'Variable:/Surface:VACANT')
theSimulator.createEntity('Variable', 'Variable:/Surface:A').Value = 15300
theSimulator.createEntity('Variable', 'Variable:/Surface:Ac').Value = 250
populator = theSimulator.createEntity('MoleculePopulateProcess', 'Process:/:populate')
populator.VariableReferenceList = [['_', 'Variable:/Surface:A'], ['_', 'Variable:/Surface:Ac']]
diffuser = theSimulator.createEntity('PeriodicBoundaryDiffusionProcess', 'Process:/:diffuse')
diffuser.VariableReferenceList = [['_', 'Variable:/Surface:A']]
diffuser.D = 4.3e-3
binder = theSimulator.createEntity('DiffusionInfluencedReactionProcess', 'Process:/:Reaction1')
binder.VariableReferenceList = [['_', 'Variable:/Surface:A','-1']]
binder.VariableReferenceList = [['_', 'Variable:/Surface:A','-1']]
binder.VariableReferenceList = [['_', 'Variable:/Surface:Ac','1']]
binder.VariableReferenceList = [['_', 'Variable:/Surface:Ac','1']]
binder.p = p1
binder = theSimulator.createEntity('DiffusionInfluencedReactionProcess', 'Process:/:Reaction2')
binder.VariableReferenceList = [['_', 'Variable:/Surface:A','-1']]
binder.VariableReferenceList = [['_', 'Variable:/Surface:Ac','-1']]
binder.VariableReferenceList = [['_', 'Variable:/Surface:Ac','1']]
binder.VariableReferenceList = [['_', 'Variable:/Surface:Ac','1']]
binder.p = p2
uni = theSimulator.createEntity('SpatiocyteNextReactionProcess', 'Process:/:Reaction3')
uni.VariableReferenceList = [['_', 'Variable:/Surface:Ac','-1']]
uni.VariableReferenceList = [['_', 'Variable:/Surface:A','1']]
uni.k = k
run(100000)
