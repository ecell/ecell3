#!/usr/bin/python2

from ecssupport import *

ENTITY     = 1
VARIABLE  = 2
PROCESS    = 3
SYSTEM     = 4

aMainWindow.printMessage("load rule\n")

#  aMainWindow.printMessage("make systems...\n")
#  aMainWindow.printMessage("\n")
#  aMainWindow.printMessage("/ --- CELL --- CYTOPLASM\n")
#  aMainWindow.printMessage("   |        |- MEMBRANE\n")
#  aMainWindow.printMessage("   -- ENVIRONMENT")

aMainWindow.theSimulator.createEntity( 'System', ( SYSTEM, '/', 'CELL' ), 'cell' )
aMainWindow.theSimulator.createEntity( 'System', ( SYSTEM, '/CELL', 'CYTOPLASM' ), 'cytoplasm' )
aMainWindow.theSimulator.createEntity( 'System', ( SYSTEM, '/CELL', 'MEMBRANE' ), 'membrane' )
aMainWindow.theSimulator.createEntity( 'System', ( SYSTEM, '/', 'ENVIRONMENT' ), 'environment' )

aMainWindow.printMessage('make variables...')
aMainWindow.theSimulator.createEntity( 'Variable', ( VARIABLE, '/', 'A' ), 'variable A' )
aMainWindow.theSimulator.createEntity( 'Variable', ( VARIABLE, '/', 'B' ), 'variable B' )
aMainWindow.theSimulator.createEntity( 'Variable', ( VARIABLE, '/', 'C' ), 'variable C' )

#  aMainWindow.theSimulator.createEntity( 'Variable', ( VARIABLE, '/CELL/CYTOPLASM', 'ATP' ), 'variable ATP' )
#  aMainWindow.theSimulator.createEntity( 'Variable', ( VARIABLE, '/CELL/CYTOPLASM', 'ADP' ), 'variable ADP' )
#  aMainWindow.theSimulator.createEntity( 'Variable', ( VARIABLE, '/CELL/CYTOPLASM', 'AMP' ), 'variable AMP' )

#  aMainWindow.theSimulator.createEntity( 'Variable', ( VARIABLE, '/ENVIRONMENT', 'GLU' ), 'Glucose' )
#  aMainWindow.theSimulator.createEntity( 'Variable', ( VARIABLE, '/ENVIRONMENT', 'PYR' ), 'Pyruvate' )
#  aMainWindow.theSimulator.createEntity( 'Variable', ( VARIABLE, '/ENVIRONMENT', 'LCT' ), 'Lactate' )

#  aMainWindow.theSimulator.createEntity( 'Variable', ( VARIABLE, '/CELL/MEMBRANE', 'CI' ), 'Channel 1' )
#  aMainWindow.theSimulator.createEntity( 'Variable', ( VARIABLE, '/CELL/MEMBRANE', 'CII' ), 'Channel 2' )
#  aMainWindow.theSimulator.createEntity( 'Variable', ( VARIABLE, '/CELL/MEMBRANE', 'CIII' ), 'Channel 3' )


aMainWindow.printMessage("make processs...\n")
try:
    aMainWindow.theSimulator.createEntity('ConstantActivityProcess',
                                         ( PROCESS, '/CELL/CYTOPLASM', 'RC1' ),
                                         'constant process' )
except:
    print 'cannot instantiate ConstantActivityProcess'
   
#  print 'set Variable:/CELL/CYTOPLASM:ATP Value = 30'
#  aMainWindow.theSimulator.setEntityProperty( ( VARIABLE, '/CELL/CYTOPLASM', 'ATP', 'Value' ), (30,) )

aMainWindow.printMessage("initialize()...\n")
aMainWindow.theSimulator.initialize()

printAllProperties( aMainWindow.theSimulator, ( SYSTEM, '', '/' ) )
printAllProperties( aMainWindow.theSimulator, ( SYSTEM, '/', 'CYTOPLASM' ) )
printProperty( aMainWindow.theSimulator, ( VARIABLE, '/', 'A', 'Value' ) )
