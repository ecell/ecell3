
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


aSimulator.createEntity( 'Substance', 'Substance:/CELL/CYTOPLASM:S', 'substance S' )
aSimulator.createEntity( 'Substance', 'Substance:/CELL/CYTOPLASM:P', 'substance P' )
aSimulator.createEntity( 'Substance', 'Substance:/CELL/CYTOPLASM:E', 'substance E' )



#aSimulator.createEntity( 'MichaelisUniUniReactor',
#                         'Reactor:/CELL/CYTOPLASM:R',
#                         'michaelis-menten reactor' )

aSimulator.createEntity( 'MassActionReactor',
                         'Reactor:/CELL/CYTOPLASM:R',
                         'a reactor' )

aSimulator.setProperty( 'Reactor:/CELL/CYTOPLASM:R:Reactant', ( 'S0', 'Substance:/CELL/CYTOPLASM:S',-1 ) )
aSimulator.setProperty( 'Reactor:/CELL/CYTOPLASM:R:Reactant', ( 'P0', 'Substance:/CELL/CYTOPLASM:P',1) )
aSimulator.setProperty( 'Reactor:/CELL/CYTOPLASM:R:Reactant', ( 'C0', 'Substance:/CELL/CYTOPLASM:E',0) )
aSimulator.setProperty( 'Reactor:/CELL/CYTOPLASM:R:K', (.00001, ) )
#aSimulator.setProperty( 'Reactor:/CELL/CYTOPLASM:R:KmS', (.01, ) )
#aSimulator.setProperty( 'Reactor:/CELL/CYTOPLASM:R:KcF', (10, ) )

 

aSimulator.setProperty( 'Substance:/CELL/CYTOPLASM:S:Quantity', (1000,) )
aSimulator.setProperty( 'Substance:/CELL/CYTOPLASM:P:Quantity', (1020,) )
aSimulator.setProperty( 'Substance:/CELL/CYTOPLASM:E:Quantity', (1400,) )

aSimulator.initialize()

printAllProperties( aSimulator, 'System::/' )
printAllProperties( aSimulator, 'System:/:CELL' )
printAllProperties( aSimulator, 'System:/CELL:CYTOPLASM' )
printAllProperties( aSimulator, 'Reactor:/CELL/CYTOPLASM:R' )


print aSimulator.getCurrentTime()
printProperty( aSimulator, 'Substance:/CELL/CYTOPLASM:S:Quantity' )
printProperty( aSimulator, 'Substance:/CELL/CYTOPLASM:P:Quantity' )
printProperty( aSimulator, 'Substance:/CELL/CYTOPLASM:E:Quantity' )

aSimulator.run( 1000 )

print aSimulator.getCurrentTime()
printProperty( aSimulator, 'Reactor:/CELL/CYTOPLASM:R:ActivityPerSecond' )
printProperty( aSimulator, 'Substance:/CELL/CYTOPLASM:S:Quantity' )
printProperty( aSimulator, 'Substance:/CELL/CYTOPLASM:S:Concentration' )
printProperty( aSimulator, 'Substance:/CELL/CYTOPLASM:S:Activity' )
printProperty( aSimulator, 'Substance:/CELL/CYTOPLASM:S:Velocity' )
printProperty( aSimulator, 'Substance:/CELL/CYTOPLASM:P:Quantity' )


