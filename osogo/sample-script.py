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

aDriver.printProperty( ( SUBSTANCE, '/CELL/CYTOPLASM', 'S', 'Quantity' ) )
aDriver.printProperty( ( SUBSTANCE, '/CELL/CYTOPLASM', 'P', 'Quantity' ) )
aDriver.printProperty( ( SUBSTANCE, '/CELL/CYTOPLASM', 'E', 'Quantity' ) )

#print '------------------------------------a'
#aDriver.getLogger( ( SUBSTANCE, '/CELL/CYTOPLASM', 'S', 'Quantity' ) )
#aDriver.getLogger( ( SUBSTANCE, '/CELL/CYTOPLASM', 'P', 'Quantity' ) )
#print aDriver.getLoggerList()
#aDriver.initialize()
#print '------------------------------------b'

print '----------------- LoggerList -------------------'
print aDriver.getLoggerList()
print '------------------------------------'

loggerS = aDriver.getLogger( ( SUBSTANCE, '/CELL/CYTOPLASM', 'S', 'Quantity' ) )
loggerP = aDriver.getLogger( ( SUBSTANCE, '/CELL/CYTOPLASM', 'P', 'Quantity' ) )
loggerE = aDriver.getLogger( ( SUBSTANCE, '/CELL/CYTOPLASM', 'E', 'Quantity' ) )

print '----------------- LoggerList -------------------'
print aDriver.getLoggerList()
print '------------------------------------'

aSession.run( 100 )

print aDriver.getCurrentTime()

aDriver.printProperty( ( SUBSTANCE, '/CELL/CYTOPLASM', 'S', 'Quantity' ) )
aDriver.printProperty( ( SUBSTANCE, '/CELL/CYTOPLASM', 'P', 'Quantity' ) )
aDriver.printProperty( ( SUBSTANCE, '/CELL/CYTOPLASM', 'E', 'Quantity' ) )

aSession.run( 100 )

print aDriver.getCurrentTime()

aDriver.printProperty( ( SUBSTANCE, '/CELL/CYTOPLASM', 'S', 'Quantity' ) )
aDriver.printProperty( ( SUBSTANCE, '/CELL/CYTOPLASM', 'P', 'Quantity' ) )
aDriver.printProperty( ( SUBSTANCE, '/CELL/CYTOPLASM', 'E', 'Quantity' ) )

print loggerS.getData()
