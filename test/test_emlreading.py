## -----------------------------------------------------
##
##
##   read simple EML file with emllib
##
##
## -----------------------------------------------------

__author__ = 'Ryosuke Suzuki'
__email__  = 'suzuki@e-cell.org'



import eml


aFile = 'readtest.eml'
aFileObject = open( aFile )
anEml = eml.Eml( aFileObject )


print '------------------------------------------------------------'

## getStepperList
print '## TEST for getStepperList'
aStepperList = anEml.getStepperList()
print aStepperList, '\n'

## 
print '## TEST for getStepperPropertyList'
aStepperPropertyList = anEml.getStepperPropertyList( 'SRM_01' )
print aStepperPropertyList, '\n'


## 
print '## TEST for getStepperProperty'
aStepperProperty = anEml.getStepperProperty( 'SRM_01', 'Hoge2' )
print aStepperProperty, '\n'


## 
print '## TEST for getStepperClass'
aStepperClass = anEml.getStepperClass( 'SRM_01' )
print aStepperClass, '\n'


print '------------------------------------------------------------'


## 
anEntityType = 'System'
aSystemPath  = '/CELL'
print '## TEST for getEntityList ( EntityType: System, SystemPath: /CELL )'
anEntityList = anEml.getEntityList( anEntityType, aSystemPath )
print anEntityList,'\n'



## 
anEntityType = 'Variable'
aSystemPath  = '/CELL/CYTOPLASM'
print '## TEST for getEntityList ( EntityType: Variable, SystemPath: /CELL/CYTOPLASM )'
anEntityList = anEml.getEntityList( anEntityType, aSystemPath )
print anEntityList, '\n'


## 
anEntityType = 'Process'
aSystemPath  = '/CELL/CYTOPLASM'
print '## TEST for getEntityList ( EntityType: Process, SystemPath: /CELL/CYTOPLASM )'
anEntityList = anEml.getEntityList( anEntityType, aSystemPath )
print anEntityList, '\n'



## 
print '## TEST for getEntityPropertyList (Variable S)'
anEntityPropertyList = anEml.getEntityPropertyList( 'Variable:/CELL/CYTOPLASM:S' )
print anEntityPropertyList, '\n'

## 
print '## TEST for getEntityPropertyList (Process E)'
anEntityPropertyList = anEml.getEntityPropertyList( 'Process:/CELL/CYTOPLASM:E' )
print anEntityPropertyList, '\n'

## 
print '## TEST for getEntityProperty (Variable S, Value)'
anEntityProperty = anEml.getEntityProperty( 'Variable:/CELL/CYTOPLASM:S', 'Value' )
print anEntityProperty, '\n'

## 
print '## TEST for getEntityProperty (VariableReference E, VariableReference)'
anEntityProperty = anEml.getEntityProperty( 'Process:/CELL/CYTOPLASM:E', 'VariableReference' )
print anEntityProperty, '\n'


