#!/usr/bin/python2

from ecssupport import *

ENTITY     = 1
SUBSTANCE  = 2
REACTOR    = 3
SYSTEM     = 4

aMainWindow.theSimulator.createEntity( 'System', ( SYSTEM, '/', 'CELL' ), 'cell' )
aMainWindow.theSimulator.createEntity( 'System', ( SYSTEM, '/CELL', 'CYTOPLASM' ), 'cytoplasm' )
aMainWindow.theSimulator.createEntity( 'System', ( SYSTEM, '/CELL', 'MEMBRANE' ), 'membrane' )
aMainWindow.theSimulator.createEntity( 'System', ( SYSTEM, '/', 'ENVIRONMENT' ), 'environment' )

print 'make substances...'
aMainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/', 'A' ), 'substance A' )
aMainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/', 'B' ), 'substance B' )
aMainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/', 'C' ), 'substance C' )

#  aMainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/CYTOPLASM', 'ATP' ), 'substance ATP' )
#  aMainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/CYTOPLASM', 'ADP' ), 'substance ADP' )
#  aMainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/CYTOPLASM', 'AMP' ), 'substance AMP' )

#  aMainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/ENVIRONMENT', 'GLU' ), 'Glucose' )
#  aMainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/ENVIRONMENT', 'PYR' ), 'Pyruvate' )
#  aMainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/ENVIRONMENT', 'LCT' ), 'Lactate' )

#  aMainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/MEMBRANE', 'CI' ), 'Channel 1' )
#  aMainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/MEMBRANE', 'CII' ), 'Channel 2' )
#  aMainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/MEMBRANE', 'CIII' ), 'Channel 3' )


print 'make reactors...'
try:
    aMainWindow.theSimulator.createEntity('ConstantActivityReactor',
                                         ( REACTOR, '/CELL/CYTOPLASM', 'RC1' ),
                                         'constant reactor' )
except:
    print 'cannot instantiate ConstantActivityReactor'
   
#  print 'set Substance:/CELL/CYTOPLASM:ATP Quantity = 30'
#  aMainWindow.theSimulator.setEntityProperty( ( SUBSTANCE, '/CELL/CYTOPLASM', 'ATP', 'Quantity' ), (30,) )

print 'initialize()...'
aMainWindow.theSimulator.initialize()


printAllProperties( aMainWindow.theSimulator, ( SYSTEM, '', '/' ) )
printAllProperties( aMainWindow.theSimulator, ( SYSTEM, '/', 'CYTOPLASM' ) )
printProperty( aMainWindow.theSimulator, ( SUBSTANCE, '/', 'A', 'Quantity' ) )
