#!/usr/bin/python2

from ecssupport import *

aSimulator = aMainWindow.theSimulator

aMainWindow.printMessage("Load Rule...\n")

aSimulator.createEntity( 'System', ( SYSTEM, '/', 'CELL' ), 'cell' )
aSimulator.createEntity( 'System', ( SYSTEM, '/CELL', 'CYTOPLASM' ), 'cytoplasm' )
aSimulator.createEntity( 'System', ( SYSTEM, '/CELL', 'MEMBRANE' ), 'membrane' )
aSimulator.createEntity( 'System', ( SYSTEM, '/', 'ENVIRONMENT' ), 'environment' )

# aSimulator.setEntityProperty( ( SYSTEM, '/CELL', 'CYTOPLASM', 'VolumeIndex' ), (1234,) )
# aSimulator.setEntityProperty( ( SYSTEM, '/', 'ENVIRONMENT', 'VolumeIndex' ), (9876,) )

aMainWindow.printMessage('make Variables...\n')

aSimulator.createEntity( 'Variable', ( VARIABLE, '/CELL/CYTOPLASM', 'ATP' ), 'variable ATP' )
aSimulator.createEntity( 'Variable', ( VARIABLE, '/CELL/CYTOPLASM', 'ADP' ), 'variable ADP' )
aSimulator.createEntity( 'Variable', ( VARIABLE, '/CELL/CYTOPLASM', 'AMP' ), 'variable AMP' )

aSimulator.createEntity( 'Variable', ( VARIABLE, '/ENVIRONMENT', 'GLU' ), 'Glucose' )
aSimulator.createEntity( 'Variable', ( VARIABLE, '/ENVIRONMENT', 'PYR' ), 'Pyruvate' )
aSimulator.createEntity( 'Variable', ( VARIABLE, '/ENVIRONMENT', 'LCT' ), 'Lactate' )

aSimulator.createEntity( 'Variable', ( VARIABLE, '/CELL/MEMBRANE', 'CI' ), 'Channel 1' )
aSimulator.createEntity( 'Variable', ( VARIABLE, '/CELL/MEMBRANE', 'CII' ), 'Channel 2' )
aSimulator.createEntity( 'Variable', ( VARIABLE, '/CELL/MEMBRANE', 'CIII' ), 'Channel 3' )


aMainWindow.printMessage("make Processs...\n")

try:
    aSimulator.createEntity('ConstantActivityProcess',
                                         ( PROCESS, '/CELL/CYTOPLASM', 'RC' ),
                                         'constant process' )
except:
   aMainWindow.printMessage("cannot instantiate ConstantActivityProcess\n")

try:
    aSimulator.createEntity('MichaelisUniUniProcess',
                                         ( PROCESS, '/CELL/CYTOPLASM', 'RM1' ),
                                         'michaelis-menten process' )
except:
   aMainWindow.printMessage("cannot instantiate MichaelisUniUniProcess\n")

 
aMainWindow.printMessage("set Value...\n")

aSimulator.setEntityProperty( ( VARIABLE, '/CELL/CYTOPLASM', 'ATP', 'Value' ), (100,) )
aSimulator.setEntityProperty( ( VARIABLE, '/CELL/CYTOPLASM', 'ADP' , 'Value' ), (120,) )
aSimulator.setEntityProperty( ( VARIABLE, '/CELL/CYTOPLASM', 'AMP', 'Value' ), (140,) )

aSimulator.setEntityProperty( ( VARIABLE, '/ENVIRONMENT', 'GLU', 'Value' ), (300,) )
aSimulator.setEntityProperty( ( VARIABLE, '/ENVIRONMENT', 'PYR', 'Value' ), (400,) )
aSimulator.setEntityProperty( ( VARIABLE, '/ENVIRONMENT', 'LCT', 'Value' ), (500,) )

aSimulator.setEntityProperty( ( VARIABLE, '/CELL/MEMBRANE', 'CI', 'Value' ), (10,) )
aSimulator.setEntityProperty( ( VARIABLE, '/CELL/MEMBRANE', 'CII', 'Value' ), (20,) )
aSimulator.setEntityProperty( ( VARIABLE, '/CELL/MEMBRANE', 'CIII', 'Value' ), (30,) )

aMainWindow.printMessage("initialize()...\n")
aSimulator.initialize()

#  aMainWindow.printMessage( '---------------------------------------\n' )
#  aMainWindow.printAllProperties( ( SYSTEM, '', '/' ) )
#  aMainWindow.printMessage( '---------------------------------------\n' )
#  aMainWindow.printAllProperties( ( SYSTEM, '/CELL', 'CYTOPLASM' ) )
#  aMainWindow.printMessage( '---------------------------------------\n' )


aMainWindow.printProperty( ( VARIABLE, '/CELL/CYTOPLASM', 'ATP', 'Value' ) )
aMainWindow.printProperty( ( VARIABLE, '/CELL/CYTOPLASM', 'ADP', 'Value' ) )
aMainWindow.printProperty( ( VARIABLE, '/CELL/CYTOPLASM', 'AMP', 'Value' ) )

aMainWindow.printProperty( ( VARIABLE, '/ENVIRONMENT', 'GLU', 'Value' ) )
aMainWindow.printProperty( ( VARIABLE, '/ENVIRONMENT', 'PYR', 'Value' ) )
aMainWindow.printProperty( ( VARIABLE, '/ENVIRONMENT', 'LCT', 'Value' ) )

aMainWindow.printProperty( ( VARIABLE, '/CELL/MEMBRANE', 'CI', 'Value' ) )
aMainWindow.printProperty( ( VARIABLE, '/CELL/MEMBRANE', 'CII', 'Value' ) )
aMainWindow.printProperty( ( VARIABLE, '/CELL/MEMBRANE', 'CIII', 'Value' ) )







