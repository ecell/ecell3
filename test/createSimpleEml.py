
# -----------------------------------------------------
#
#
#   create simple EML file with emllib
#
#
# -----------------------------------------------------
#
# [usage]
#
#  % /path/to/ecell3 createSimpleEml.py
#
#  A new file 'output.eml' is made.
#
#  (to make indentations, use nmaker program)
#
# -----------------------------------------------------

__author__ = 'Ryosuke Suzuki'
__email__  = 'suzuki@e-cell.org'



import eml

aFile = 'minimum.eml'

aFileObject = open( aFile, 'w' )
aFileObject.write( '<?xml version="1.0"?><eml></eml>' )

aFile = 'minimum.eml'
aFileObject = open( aFile )
anEml = eml.Eml( aFileObject )


## asString TEST
#string = anEml.asString()
#print string


## save TEST
#anEml.save( 'newFile.eml' )


## createStepper TEST
anEml.createStepperlist()
anEml.createStepper( 'Euler1SRMStepper', 'SRM_01', ( ) )


## createEntity [System] TEST
anEml.createEntity( 'System', 'System', 'System::/', 'environment' )
anEml.createEntity( 'System', 'System', 'System:/:CELL', 'cell' )
anEml.createEntity( 'System', 'System', 'System:/CELL:CYTOPLASM', 'cytoplasm' )

anEml.setProperty( 'System::/',              'Volume', [ '0.000000000000001'] )
anEml.setProperty( 'System:/:CELL',          'Volume', [ 'unknown'] )
anEml.setProperty( 'System:/CELL:CYTOPLASM', 'Volume', [ '1e-18'] )

anEml.setProperty( 'System::/',              'StepperID', [ 'SRM_01' ] )
anEml.setProperty( 'System:/:CELL',          'StepperID', [ 'SRM_01' ] )
anEml.setProperty( 'System:/CELL:CYTOPLASM', 'StepperID', [ 'SRM_01' ] )



## createEntity [Variable] TEST
anEml.createEntity( 'Variable', 'SRMVariable', 'Variable:/CELL/CYTOPLASM:S', 'variable S' )
anEml.createEntity( 'Variable', 'SRMVariable', 'Variable:/CELL/CYTOPLASM:P', 'variable P' )
anEml.createEntity( 'Variable', 'SRMVariable', 'Variable:/CELL/CYTOPLASM:E', 'variable E' )

anEml.setProperty( 'Variable:/CELL/CYTOPLASM:S', 'Value', [ '1000' ] )
anEml.setProperty( 'Variable:/CELL/CYTOPLASM:P', 'Value', [ '0' ] )
anEml.setProperty( 'Variable:/CELL/CYTOPLASM:E', 'Value', [ '1400' ] )


## createEntity [Process] TEST
anEml.createEntity( 'Process', 'MichaelisUniUniProcess', 'Process:/CELL/CYTOPLASM:E', 'process E' )


anEml.setProperty( 'Process:/CELL/CYTOPLASM:E', 'VariableReference', \
                   ( 'S0', 'Variable:/CELL/CYTOPLASM:S', '-1' ) )
anEml.setProperty( 'Process:/CELL/CYTOPLASM:E','VariableReference', \
                   ( 'P0', 'Variable:/CELL/CYTOPLASM:P', '1') )
anEml.setProperty( 'Process:/CELL/CYTOPLASM:E','VariableReference', \
                   ( 'C0', 'Variable:/CELL/CYTOPLASM:E', '0') )

anEml.setProperty( 'Process:/CELL/CYTOPLASM:E','KmS', [ '1' ] )
anEml.setProperty( 'Process:/CELL/CYTOPLASM:E','KcF', [ '10' ] )

anEml.save( 'output.eml' )

#string = anEml.asString()
#print string



### ToDo ###
## redundancy of entity -> system, variable and process, ok!
## redundancy of Property -> system, variable almost ok! process???
##  error for no parents
