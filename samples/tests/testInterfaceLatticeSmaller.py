sim = theSimulator.createStepper('SpatiocyteStepper', 'SS')
sim.VoxelRadius = 0.74e-9
sim.SearchVacant = 0

theSimulator.rootSystem.StepperID = 'SS'
theSimulator.createEntity('Variable', 'Variable:/:GEOMETRY').Value = 0
theSimulator.createEntity('Variable', 'Variable:/:LENGTHX').Value = 1e-7
theSimulator.createEntity('Variable', 'Variable:/:LENGTHY').Value = 1e-7
theSimulator.createEntity('Variable', 'Variable:/:LENGTHZ').Value = 1e-7
theSimulator.createEntity('Variable', 'Variable:/:VACANT')
theSimulator.createEntity('Variable', 'Variable:/:XYPLANE').Value = 5
theSimulator.createEntity('Variable', 'Variable:/:XZPLANE').Value = 5
theSimulator.createEntity('Variable', 'Variable:/:YZPLANE').Value = 3

# Create the surface compartment:
theSimulator.createEntity('System', 'System:/:Surface').StepperID = 'SS'
theSimulator.createEntity('Variable', 'Variable:/Surface:DIMENSION').Value = 2
theSimulator.createEntity('Variable', 'Variable:/Surface:VACANT')

theSimulator.createEntity('Variable', 'Variable:/Surface:ANIO').Value = 0
theSimulator.createEntity('Variable', 'Variable:/Surface:ANIOs').Value = 0
theSimulator.createEntity('Variable', 'Variable:/Surface:ANIO_PTEN').Value = 0
theSimulator.createEntity('Variable', 'Variable:/Surface:ANIOs_PTEN').Value = 0
theSimulator.createEntity('Variable', 'Variable:/:PTEN').Value = 10000
theSimulator.createEntity('Variable', 'Variable:/:PTEN2').Value = 0
theSimulator.createEntity('Variable', 'Variable:/Surface:PTEN').Value = 20

logger = theSimulator.createEntity('VisualizationLogProcess', 'Process:/:logger')
logger.VariableReferenceList = [['_', 'Variable:/Surface:ANIO_PTEN']]
logger.VariableReferenceList = [['_', 'Variable:/Surface:ANIOs_PTEN']]
logger.VariableReferenceList = [['_', 'Variable:/Surface:PTEN']]
logger.VariableReferenceList = [['_', 'Variable:/:PTEN']]
logger.VariableReferenceList = [['_', 'Variable:/:Vacant']]
logger.VariableReferenceList = [['_', 'Variable:/Surface:VACANT']]
logger.VariableReferenceList = [['_', 'Variable:/:Interface']]
logger.VariableReferenceList = [['_', 'Variable:/Surface:ANIO']]
logger.VariableReferenceList = [['_', 'Variable:/:PTEN2']]
logger.VariableReferenceList = [['_', 'Variable:/Surface:ANIOs']]
logger.LogInterval = 1e-5
logger.MultiscaleStructure = 0

populator = theSimulator.createEntity('MoleculePopulateProcess', 'Process:/:pop')
populator.VariableReferenceList = [['_', 'Variable:/Surface:ANIO']]
populator.VariableReferenceList = [['_', 'Variable:/Surface:ANIO_PTEN']]
populator.VariableReferenceList = [['_', 'Variable:/Surface:ANIOs_PTEN']]

populator = theSimulator.createEntity('MoleculePopulateProcess', 'Process:/:pop2')
populator.VariableReferenceList = [['_', 'Variable:/Surface:PTEN']]
populator.UniformRadiusY = 0.99
populator.UniformRadiusZ = 0.99

populator = theSimulator.createEntity('MoleculePopulateProcess', 'Process:/:pop3')
populator.VariableReferenceList = [['_', 'Variable:/:PTEN']]
populator.OriginX = -1.5
populator.OriginY = 1.5

react = theSimulator.createEntity('SpatiocyteNextReactionProcess', 'Process:/:desorp')
react.VariableReferenceList = [['_', 'Variable:/Surface:PTEN','-1']]
react.VariableReferenceList = [['_', 'Variable:/:PTEN2','1']]
react.SearchVacant = 1
react.k = 1e+5

diffuser = theSimulator.createEntity('DiffusionProcess', 'Process:/:diffusePTENv')
diffuser.VariableReferenceList = [['_', 'Variable:/:PTEN']]
diffuser.D = 5e-12

diffuser = theSimulator.createEntity('DiffusionProcess', 'Process:/:diffusePTEN2v')
diffuser.VariableReferenceList = [['_', 'Variable:/:PTEN2']]
diffuser.D = 5e-12

fil = theSimulator.createEntity('CompartmentProcess', 'Process:/:filam')
fil.VariableReferenceList = [['_', 'Variable:/Surface:PTEN']]
fil.OriginX = 0
fil.OriginZ = -0.4
fil.RotateZ = 0.78
fil.Length = 1.8e-7
fil.Width = 1.8e-7
fil.Autofit = 0
fil.DiffuseRadius = 0.8e-8
fil.LipidRadius = 0.8e-8
fil.Periodic = 0
fil.RegularLattice = 1
fil.SurfaceDirection = 2

import time
run(1e-6)
print "Done stirring. Now running..."
start = time.time()
run(0.001)
end = time.time()
duration = end-start
print duration
