#! /usr/bin/python2

import ecs

from ecssupport import *

import Session
from samplerule import *

aSession = Session.SingleSession( 'tmp.py' )
aDriver = aSession.theDriver
aModelInterpreter = aSession.theModelInterpreter

aModelInterpreter.load( aCellModelObject )
aDriver.initialize()

aDriver.printProperty( ( VARIABLE, '/CELL/CYTOPLASM', 'S', 'Value' ) )
aDriver.printProperty( ( VARIABLE, '/CELL/CYTOPLASM', 'P', 'Value' ) )
aDriver.printProperty( ( VARIABLE, '/CELL/CYTOPLASM', 'E', 'Value' ) )

#print '------------------------------------a'
#aDriver.getLogger( ( VARIABLE, '/CELL/CYTOPLASM', 'S', 'Value' ) )
#aDriver.getLogger( ( VARIABLE, '/CELL/CYTOPLASM', 'P', 'Value' ) )
#print aDriver.getLoggerList()
#aDriver.initialize()
#print '------------------------------------b'

print '----------------- LoggerList -------------------'
print aDriver.getLoggerList()
print '------------------------------------'

loggerS = aDriver.getLogger( ( VARIABLE, '/CELL/CYTOPLASM', 'S', 'Value' ) )
loggerP = aDriver.getLogger( ( VARIABLE, '/CELL/CYTOPLASM', 'P', 'Value' ) )
loggerE = aDriver.getLogger( ( VARIABLE, '/CELL/CYTOPLASM', 'E', 'Value' ) )

print '----------------- LoggerList -------------------'
print aDriver.getLoggerList()
print '------------------------------------'

print 'start = ', loggerS.getStartTime()
print 'end = ', loggerS.getEndTime()

aSession.run( 100 )

print aDriver.getCurrentTime()

aDriver.printProperty( ( VARIABLE, '/CELL/CYTOPLASM', 'S', 'Value' ) )
aDriver.printProperty( ( VARIABLE, '/CELL/CYTOPLASM', 'P', 'Value' ) )
aDriver.printProperty( ( VARIABLE, '/CELL/CYTOPLASM', 'E', 'Value' ) )

aSession.run( 100 )

print aDriver.getCurrentTime()

aDriver.printProperty( ( VARIABLE, '/CELL/CYTOPLASM', 'S', 'Value' ) )
aDriver.printProperty( ( VARIABLE, '/CELL/CYTOPLASM', 'P', 'Value' ) )
aDriver.printProperty( ( VARIABLE, '/CELL/CYTOPLASM', 'E', 'Value' ) )

#print loggerS.getData()

print 'start = ', loggerS.getStartTime()
print 'end = ', loggerS.getEndTime()




