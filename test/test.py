import ecell.Session
import ecell.ecs

from ecell.FullID import *

from ecell.util import *
from ecell.ECS import *


print 'create a Session'
aSession = ecell.Session.Session( ecell.ecs.Simulator() )
aSimulator = aSession.theSimulator

aSimulator.createStepper( 'RungeKutta4SRMStepper', 'RK4_0' )
aSimulator.createStepper( 'RungeKutta4SRMStepper', 'RK4_1' )
aSimulator.createStepper( 'Euler1SRMStepper', 'E1_0' )

aSimulator.setProperty( 'System::/:StepperID', ('RK4_0',) )



print 'make substances...'
aSimulator.createEntity( 'Substance', 'Substance:/:A', 'substance A' )
aSimulator.createEntity( 'Substance', 'Substance:/:B', 'substance B' )

aSimulator.run(10)

print 'initialize()...'
aSimulator.initialize()

aSimulator.run(10)

aSimulator.createEntity( 'System', 'System:/:CYTOPLASM', 'cytoplasm' )
aSimulator.setProperty( 'System:/:CYTOPLASM:StepperID', ('RK4_0',) )


aSimulator.createEntity( 'Substance', 'Substance:/CYTOPLASM:CA', 's CA' )
aSimulator.createEntity( 'Substance', 'Substance:/CYTOPLASM:CB', 's CB' )


print 'initialize()...'
aSimulator.initialize()


print 'set Substance:/:A Quantity = 30'
aSimulator.setProperty( 'Substance:/:A:Quantity', (30,) )





printAllProperties( aSimulator, 'System::/' )

printAllProperties( aSimulator, 'System:/:CYTOPLASM' )


substancelist = aSimulator.getProperty( 'System::/:SubstanceList' )

for i in substancelist:
    printAllProperties( aSimulator, 'Substance:/:' + i )

substancelist = aSimulator.getProperty( 'System:/:CYTOPLASM:SubstanceList' )

for i in substancelist:
    printAllProperties( aSimulator, 'Substance:/CYTOPLASM:' + i )


print

printProperty( aSimulator, 'Substance:/:A:Quantity' )

#printProperty( aSimulator, 'Substance:/:A:Quantity' )
print 'changing Quantity of Substance:/:A...'
aSimulator.setProperty( 'Substance:/:A:Quantity', (10.0, ) )
printProperty( aSimulator, 'Substance:/:A:Quantity' )


print 'step()...'
print aSimulator.getCurrentTime()
aSimulator.step()

print aSimulator.getCurrentTime()
aSimulator.step()

print aSimulator.getCurrentTime()
aSimulator.step()

print aSimulator.getCurrentTime()
aSimulator.step()
