from Numeric import *

import sys

import ecell.Session
import ecell.ecs

from ecell.util import *
from ecell.ECS import *
from ecell.eml import *

print 'create a simulation session instance'
aSession = ecell.Session.Session( ecell.ecs.Simulator() )
aSimulator = aSession.theSimulator

aFile = open( sys.argv[1] )

print 'load Model'
aSession.loadModel( aFile )

aFile.close()

print 'init'
aSimulator.initialize()

aLogger1 = LoggerStub( aSimulator, 'Substance:/CELL/CYTOPLASM:S:Quantity'  )
aLogger1.create()

aSubstanceS = EntityStub( aSimulator, 'Substance:/CELL/CYTOPLASM:S' )

print 'Current time = ', aSimulator.getCurrentTime()
print aSubstanceS.theFullIDString, ':Quantity =', aSubstanceS.getProperty( 'Quantity' )
printProperty( aSimulator, 'Substance:/CELL/CYTOPLASM:P:Quantity' )
printProperty( aSimulator, 'Substance:/CELL/CYTOPLASM:E:Quantity' )


aSimulator.run( 1000 )

print 'Current time = ', aSimulator.getCurrentTime()
print aSubstanceS.theFullIDString, ':'
print '\tQuantity =', aSubstanceS.getProperty( 'Quantity' )
print '\tConcentration =', aSubstanceS.getProperty( 'Concentration' )
print '\tTotalVelocity =', aSubstanceS.getProperty( 'TotalVelocity' )
printProperty( aSimulator, 'Substance:/CELL/CYTOPLASM:P:Quantity' )


print 'logger list:', aSimulator.getLoggerList()

if aLogger1:
#if 0:
    print "Logger: start: %s, end: %s, size: %s" %\
          ( aLogger1.getStartTime(), aLogger1.getEndTime(),\
            aLogger1.getSize() )
    print aLogger1.getData( 0, aLogger1.getEndTime() )[:5]
    print aLogger1.getData( aLogger1.getEndTime() - 10 ,
                            aLogger1.getEndTime() , .5 )[:5]
    print aLogger1.getData()[-10:]


