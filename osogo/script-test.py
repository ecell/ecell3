#!/usr/bin/python2

ENTITY     = 1
SUBSTANCE  = 2
REACTOR    = 3
SYSTEM     = 4

MainWindow.theSimulator.createEntity( 'System', ( SYSTEM, '/', 'CELL' ), 'cell' )
MainWindow.theSimulator.createEntity( 'System', ( SYSTEM, '/CELL', 'CYTOPLASM' ), 'cytoplasm' )
MainWindow.theSimulator.createEntity( 'System', ( SYSTEM, '/CELL', 'MEMBRANE' ), 'membrane' )
MainWindow.theSimulator.createEntity( 'System', ( SYSTEM, '/', 'ENVIRONMENT' ), 'environment' )

print 'make substances...'
MainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/', 'A' ), 'substance A' )
MainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/', 'B' ), 'substance B' )
MainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/', 'C' ), 'substance C' )

#  MainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/CYTOPLASM', 'ATP' ), 'substance ATP' )
#  MainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/CYTOPLASM', 'ADP' ), 'substance ADP' )
#  MainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/CYTOPLASM', 'AMP' ), 'substance AMP' )

#  MainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/ENVIRONMENT', 'GLU' ), 'Glucose' )
#  MainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/ENVIRONMENT', 'PYR' ), 'Pyruvate' )
#  MainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/ENVIRONMENT', 'LCT' ), 'Lactate' )

#  MainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/MEMBRANE', 'CI' ), 'Channel 1' )
#  MainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/MEMBRANE', 'CII' ), 'Channel 2' )
#  MainWindow.theSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/MEMBRANE', 'CIII' ), 'Channel 3' )


print 'make reactors...'
try:
    MainWindow.theSimulator.createEntity('ConstantActivityReactor',
                                         ( REACTOR, '/CELL/CYTOPLASM', 'RC1' ),
                                         'constant reactor' )
except:
    print 'cannot instantiate ConstantActivityReactor'
   
#  print 'set Substance:/CELL/CYTOPLASM:ATP Quantity = 30'
#  MainWindow.theSimulator.setProperty( ( SUBSTANCE, '/CELL/CYTOPLASM', 'ATP', 'Quantity' ), (30,) )

print 'initialize()...'
MainWindow.theSimulator.initialize()
