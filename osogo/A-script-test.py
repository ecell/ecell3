#!/usr/bin/python2

from ecssupport import *

ENTITY     = 1
SUBSTANCE  = 2
REACTOR    = 3
SYSTEM     = 4

aSimulator = aMainWindow.theSimulator

aMainWindow.printMessage("load rule\n")

aSimulator.createEntity( 'System', ( SYSTEM, '/', 'CELL' ), 'cell' )
aSimulator.createEntity( 'System', ( SYSTEM, '/CELL', 'CYTOPLASM' ), 'cytoplasm' )
aSimulator.createEntity( 'System', ( SYSTEM, '/CELL', 'MEMBRANE' ), 'membrane' )
aSimulator.createEntity( 'System', ( SYSTEM, '/', 'ENVIRONMENT' ), 'environment' )

aMainWindow.printMessage('make substances...')
#  aSimulator.createEntity( 'Substance', ( SUBSTANCE, '/', 'A' ), 'substance A' )
#  aSimulator.createEntity( 'Substance', ( SUBSTANCE, '/', 'B' ), 'substance B' )
#  aSimulator.createEntity( 'Substance', ( SUBSTANCE, '/', 'C' ), 'substance C' )

aSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/CYTOPLASM', 'ATP' ), 'substance ATP' )
aSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/CYTOPLASM', 'ADP' ), 'substance ADP' )
aSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/CYTOPLASM', 'AMP' ), 'substance AMP' )

aSimulator.createEntity( 'Substance', ( SUBSTANCE, '/ENVIRONMENT', 'GLU' ), 'Glucose' )
aSimulator.createEntity( 'Substance', ( SUBSTANCE, '/ENVIRONMENT', 'PYR' ), 'Pyruvate' )
aSimulator.createEntity( 'Substance', ( SUBSTANCE, '/ENVIRONMENT', 'LCT' ), 'Lactate' )

aSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/MEMBRANE', 'CI' ), 'Channel 1' )
aSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/MEMBRANE', 'CII' ), 'Channel 2' )
aSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/MEMBRANE', 'CIII' ), 'Channel 3' )


#  aMainWindow.printMessage("make reactors...\n")
#  try:
#      aSimulator.createEntity('ConstantActivityReactor',
#                                           ( REACTOR, '/CELL/CYTOPLASM', 'RC1' ),
#                                           'constant reactor' )
#  except:
#      print 'cannot instantiate ConstantActivityReactor'
   
aMainWindow.printMessage('set Substance:/CELL/CYTOPLASM:ATP Quantity = 30')
aSimulator.setProperty( ( SUBSTANCE, '/CELL/CYTOPLASM', 'ATP', 'Quantity' ), (30,) )

#  aMainWindow.printMessage("initialize()...\n")
#  aSimulator.initialize()

#  aMainWindow.printAllProperties( ( SYSTEM, '', '/' ) )
#  aMainWindow.printAllProperties( ( SYSTEM, '/', 'CYTOPLASM' ) )
#  aMainWindow.printProperty( ( SUBSTANCE, '/', 'A', 'Quantity' ) )








