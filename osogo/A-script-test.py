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

aMainWindow.printMessage('make Substances...\n')

aSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/CYTOPLASM', 'ATP' ), 'substance ATP' )
aSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/CYTOPLASM', 'ADP' ), 'substance ADP' )
aSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/CYTOPLASM', 'AMP' ), 'substance AMP' )

aSimulator.createEntity( 'Substance', ( SUBSTANCE, '/ENVIRONMENT', 'GLU' ), 'Glucose' )
aSimulator.createEntity( 'Substance', ( SUBSTANCE, '/ENVIRONMENT', 'PYR' ), 'Pyruvate' )
aSimulator.createEntity( 'Substance', ( SUBSTANCE, '/ENVIRONMENT', 'LCT' ), 'Lactate' )

aSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/MEMBRANE', 'CI' ), 'Channel 1' )
aSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/MEMBRANE', 'CII' ), 'Channel 2' )
aSimulator.createEntity( 'Substance', ( SUBSTANCE, '/CELL/MEMBRANE', 'CIII' ), 'Channel 3' )


aMainWindow.printMessage("make Reactors...\n")

try:
    aSimulator.createEntity('ConstantActivityReactor',
                                         ( REACTOR, '/CELL/CYTOPLASM', 'RC' ),
                                         'constant reactor' )
except:
   aMainWindow.printMessage("cannot instantiate ConstantActivityReactor\n")

try:
    aSimulator.createEntity('MichaelisUniUniReactor',
                                         ( REACTOR, '/CELL/CYTOPLASM', 'RM1' ),
                                         'michaelis-menten reactor' )
except:
   aMainWindow.printMessage("cannot instantiate MichaelisUniUniReactor\n")

 
aMainWindow.printMessage("set Quantity...\n")

aSimulator.setEntityProperty( ( SUBSTANCE, '/CELL/CYTOPLASM', 'ATP', 'Quantity' ), (100,) )
aSimulator.setEntityProperty( ( SUBSTANCE, '/CELL/CYTOPLASM', 'ADP' , 'Quantity' ), (120,) )
aSimulator.setEntityProperty( ( SUBSTANCE, '/CELL/CYTOPLASM', 'AMP', 'Quantity' ), (140,) )

aSimulator.setEntityProperty( ( SUBSTANCE, '/ENVIRONMENT', 'GLU', 'Quantity' ), (300,) )
aSimulator.setEntityProperty( ( SUBSTANCE, '/ENVIRONMENT', 'PYR', 'Quantity' ), (400,) )
aSimulator.setEntityProperty( ( SUBSTANCE, '/ENVIRONMENT', 'LCT', 'Quantity' ), (500,) )

aSimulator.setEntityProperty( ( SUBSTANCE, '/CELL/MEMBRANE', 'CI', 'Quantity' ), (10,) )
aSimulator.setEntityProperty( ( SUBSTANCE, '/CELL/MEMBRANE', 'CII', 'Quantity' ), (20,) )
aSimulator.setEntityProperty( ( SUBSTANCE, '/CELL/MEMBRANE', 'CIII', 'Quantity' ), (30,) )

aMainWindow.printMessage("initialize()...\n")
aSimulator.initialize()

#  aMainWindow.printMessage( '---------------------------------------\n' )
#  aMainWindow.printAllProperties( ( SYSTEM, '', '/' ) )
#  aMainWindow.printMessage( '---------------------------------------\n' )
#  aMainWindow.printAllProperties( ( SYSTEM, '/CELL', 'CYTOPLASM' ) )
#  aMainWindow.printMessage( '---------------------------------------\n' )


aMainWindow.printProperty( ( SUBSTANCE, '/CELL/CYTOPLASM', 'ATP', 'Quantity' ) )
aMainWindow.printProperty( ( SUBSTANCE, '/CELL/CYTOPLASM', 'ADP', 'Quantity' ) )
aMainWindow.printProperty( ( SUBSTANCE, '/CELL/CYTOPLASM', 'AMP', 'Quantity' ) )

aMainWindow.printProperty( ( SUBSTANCE, '/ENVIRONMENT', 'GLU', 'Quantity' ) )
aMainWindow.printProperty( ( SUBSTANCE, '/ENVIRONMENT', 'PYR', 'Quantity' ) )
aMainWindow.printProperty( ( SUBSTANCE, '/ENVIRONMENT', 'LCT', 'Quantity' ) )

aMainWindow.printProperty( ( SUBSTANCE, '/CELL/MEMBRANE', 'CI', 'Quantity' ) )
aMainWindow.printProperty( ( SUBSTANCE, '/CELL/MEMBRANE', 'CII', 'Quantity' ) )
aMainWindow.printProperty( ( SUBSTANCE, '/CELL/MEMBRANE', 'CIII', 'Quantity' ) )







