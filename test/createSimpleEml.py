
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



## createEntity [Substance] TEST
anEml.createEntity( 'Substance', 'SRMSubstance', 'Substance:/CELL/CYTOPLASM:S', 'substance S' )
anEml.createEntity( 'Substance', 'SRMSubstance', 'Substance:/CELL/CYTOPLASM:P', 'substance P' )
anEml.createEntity( 'Substance', 'SRMSubstance', 'Substance:/CELL/CYTOPLASM:E', 'substance E' )

anEml.setProperty( 'Substance:/CELL/CYTOPLASM:S', 'Quantity', [ '1000' ] )
anEml.setProperty( 'Substance:/CELL/CYTOPLASM:P', 'Quantity', [ '0' ] )
anEml.setProperty( 'Substance:/CELL/CYTOPLASM:E', 'Quantity', [ '1400' ] )


## createEntity [Reactor] TEST
anEml.createEntity( 'Reactor', 'MichaelisUniUniReactor', 'Reactor:/CELL/CYTOPLASM:E', 'reactor E' )


anEml.setProperty( 'Reactor:/CELL/CYTOPLASM:E', 'Reactant', \
                   ( 'S0', 'Substance:/CELL/CYTOPLASM:S', '-1' ) )
anEml.setProperty( 'Reactor:/CELL/CYTOPLASM:E','Reactant', \
                   ( 'P0', 'Substance:/CELL/CYTOPLASM:P', '1') )
anEml.setProperty( 'Reactor:/CELL/CYTOPLASM:E','Reactant', \
                   ( 'C0', 'Substance:/CELL/CYTOPLASM:E', '0') )

anEml.setProperty( 'Reactor:/CELL/CYTOPLASM:E','KmS', [ '1' ] )
anEml.setProperty( 'Reactor:/CELL/CYTOPLASM:E','KcF', [ '10' ] )

anEml.save( 'output.eml' )

#string = anEml.asString()
#print string



### ToDo ###
## redundancy of entity -> system, substance and reactor, ok!
## redundancy of Property -> system, substance almost ok! reactor???
##  error for no parents
