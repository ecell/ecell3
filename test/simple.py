
from ecell.util import *
import ecell.ecs

aSimulator = ecell.ecs.Simulator()
aSimulator.createStepper( 'Euler1SRMStepper', 'E1SRM_0' )

aSimulator.setProperty( 'System::/:StepperID', ('E1SRM_0',) )

aSimulator.createEntity( 'System', 'System:/:CELL', 'cell' )
aSimulator.setProperty( 'System:/:CELL:StepperID', ('E1SRM_0',) )
aSimulator.createEntity( 'System', 'System:/CELL:CYTOPLASM', 'cytoplasm' )

aSimulator.setProperty( 'System:/CELL:CYTOPLASM:StepperID', ('E1SRM_0',) )

aSimulator.createEntity( 'System', 'System:/CELL:MEMBRANE', 'membrane' )

aSimulator.setProperty( 'System:/CELL:MEMBRANE:StepperID', ('E1SRM_0',) )
aSimulator.createEntity( 'System', 'System:/:ENVIRONMENT', 'environment' )

aSimulator.setProperty( 'System:/:ENVIRONMENT:StepperID', ('E1SRM_0',) )

aSimulator.setProperty( 'System:/CELL:CYTOPLASM:Volume', (10e-17,) )


aSimulator.createEntity( 'Variable', 'Variable:/CELL/CYTOPLASM:S', 'variable S' )
aSimulator.createEntity( 'Variable', 'Variable:/CELL/CYTOPLASM:P', 'variable P' )
aSimulator.createEntity( 'Variable', 'Variable:/CELL/CYTOPLASM:E', 'variable E' )



#aSimulator.createEntity( 'MichaelisUniUniProcess',
#                         'Process:/CELL/CYTOPLASM:R',
#                         'michaelis-menten process' )

aSimulator.createEntity( 'MassActionProcess',
                         'Process:/CELL/CYTOPLASM:R',
                         'a process' )

aSimulator.setProperty( 'Process:/CELL/CYTOPLASM:R:Connection', ( 'S0', 'Variable:/CELL/CYTOPLASM:S',-1 ) )
aSimulator.setProperty( 'Process:/CELL/CYTOPLASM:R:Connection', ( 'P0', 'Variable:/CELL/CYTOPLASM:P',1) )
aSimulator.setProperty( 'Process:/CELL/CYTOPLASM:R:Connection', ( 'C0', 'Variable:/CELL/CYTOPLASM:E',0) )
aSimulator.setProperty( 'Process:/CELL/CYTOPLASM:R:K', (.00001, ) )
#aSimulator.setProperty( 'Process:/CELL/CYTOPLASM:R:KmS', (.01, ) )
#aSimulator.setProperty( 'Process:/CELL/CYTOPLASM:R:KcF', (10, ) )

 

aSimulator.setProperty( 'Variable:/CELL/CYTOPLASM:S:Value', (1000,) )
aSimulator.setProperty( 'Variable:/CELL/CYTOPLASM:P:Value', (1020,) )
aSimulator.setProperty( 'Variable:/CELL/CYTOPLASM:E:Value', (1400,) )

aSimulator.initialize()

printAllProperties( aSimulator, 'System::/' )
printAllProperties( aSimulator, 'System:/:CELL' )
printAllProperties( aSimulator, 'System:/CELL:CYTOPLASM' )
printAllProperties( aSimulator, 'Process:/CELL/CYTOPLASM:R' )


print aSimulator.getCurrentTime()
printProperty( aSimulator, 'Variable:/CELL/CYTOPLASM:S:Value' )
printProperty( aSimulator, 'Variable:/CELL/CYTOPLASM:P:Value' )
printProperty( aSimulator, 'Variable:/CELL/CYTOPLASM:E:Value' )

aSimulator.run( 1000 )

print aSimulator.getCurrentTime()
printProperty( aSimulator, 'Process:/CELL/CYTOPLASM:R:ActivityPerSecond' )
printProperty( aSimulator, 'Variable:/CELL/CYTOPLASM:S:Value' )
printProperty( aSimulator, 'Variable:/CELL/CYTOPLASM:S:Concentration' )
printProperty( aSimulator, 'Variable:/CELL/CYTOPLASM:S:Activity' )
printProperty( aSimulator, 'Variable:/CELL/CYTOPLASM:S:Velocity' )
printProperty( aSimulator, 'Variable:/CELL/CYTOPLASM:P:Value' )


