from Numeric import *

import ecell.Session
import ecell.ecs

from ecell.util import *
from ecell.ECS import *
from ecell.eml import *

print 'create a simulation session instance'
aSession = ecell.Session.Session( ecell.ecs.Simulator() )
aSimulator = aSession.theSimulator

aFile = open( 'simple.eml' )

print 'load Model'
aSession.loadModel( aFile )

aFile.close()

print 'init'
aSimulator.initialize()

aLogger1 = aSimulator.getLogger( 'Substance:/CELL/CYTOPLASM:S:Quantity'  )




#printAllProperties( aSimulator, 'System::/' )
#printAllProperties( aSimulator, 'System:/:CELL' )
#printAllProperties( aSimulator, 'System:/CELL:CYTOPLASM' )
#printAllProperties( aSimulator, 'Reactor:/CELL/CYTOPLASM:E' )


print 'Current time = ', aSimulator.getCurrentTime()
printProperty( aSimulator, 'Substance:/CELL/CYTOPLASM:S:Quantity' )
printProperty( aSimulator, 'Substance:/CELL/CYTOPLASM:P:Quantity' )
printProperty( aSimulator, 'Substance:/CELL/CYTOPLASM:E:Quantity' )


aSimulator.run( 1000 )

print 'Current time = ', aSimulator.getCurrentTime()
printProperty( aSimulator, 'Reactor:/CELL/CYTOPLASM:E:ActivityPerSecond' )
printProperty( aSimulator, 'Substance:/CELL/CYTOPLASM:S:Quantity' )
printProperty( aSimulator, 'Substance:/CELL/CYTOPLASM:S:Concentration' )
printProperty( aSimulator, 'Substance:/CELL/CYTOPLASM:S:Activity' )
printProperty( aSimulator, 'Substance:/CELL/CYTOPLASM:S:Velocity' )
printProperty( aSimulator, 'Substance:/CELL/CYTOPLASM:P:Quantity' )


print 'logger list:', aSimulator.getLoggerList()

if aLogger1:
#if 0:
    print "Logger: name: %s, start: %s, end: %s, size: %s" %\
          ( aLogger1.getName(),
            aLogger1.getStartTime(), aLogger1.getEndTime(),\
            aLogger1.getSize() )
    print aLogger1.getData( 0, aLogger1.getEndTime() )[:5]
    print aLogger1.getData( aLogger1.getEndTime() - 10 ,
                            aLogger1.getEndTime() , .5 )[:5]
    print aLogger1.getData()[-10:]


