from Numeric import *

import sys

import ecell.Session
import ecell.ecs

from ecell.util import *
from ecell.ECS import *
from ecell.eml import *

print 'create a simulation session instance'
aSession = ecell.Session( ecell.ecs.Simulator() )
aSimulator = aSession.theSimulator

aFile = open( sys.argv[1] )

print 'load Model'
aSession.loadModel( aFile )

aFile.close()

print 'init'
aSimulator.initialize()

aLogger1 = LoggerStub( aSimulator, 'Variable:/CELL/CYTOPLASM:S:Value'  )
aLogger1.create()

aVariableS = EntityStub( aSimulator, 'Variable:/CELL/CYTOPLASM:S' )

print 'Current time = ', aSimulator.getCurrentTime()
print aVariableS.theFullIDString, ':Value =', aVariableS.getProperty( 'Value' )
printProperty( aSimulator, 'Variable:/CELL/CYTOPLASM:P:Value' )
printProperty( aSimulator, 'Variable:/CELL/CYTOPLASM:E:Value' )


aSimulator.run( 1000 )

print 'Current time = ', aSimulator.getCurrentTime()
print aVariableS.theFullIDString, ':'
print '\tValue =', aVariableS.getProperty( 'Value' )
print '\tConcentration =', aVariableS.getProperty( 'Concentration' )
print '\tTotalVelocity =', aVariableS.getProperty( 'TotalVelocity' )
printProperty( aSimulator, 'Variable:/CELL/CYTOPLASM:P:Value' )


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


