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
anEntityType = 'Substance'
aSystemPath  = '/CELL/CYTOPLASM'
print '## TEST for getEntityList ( EntityType: Substance, SystemPath: /CELL/CYTOPLASM )'
anEntityList = anEml.getEntityList( anEntityType, aSystemPath )
print anEntityList, '\n'


## 
anEntityType = 'Reactor'
aSystemPath  = '/CELL/CYTOPLASM'
print '## TEST for getEntityList ( EntityType: Reactor, SystemPath: /CELL/CYTOPLASM )'
anEntityList = anEml.getEntityList( anEntityType, aSystemPath )
print anEntityList, '\n'



## 
print '## TEST for getEntityPropertyList (Substance S)'
anEntityPropertyList = anEml.getEntityPropertyList( 'Substance:/CELL/CYTOPLASM:S' )
print anEntityPropertyList, '\n'

## 
print '## TEST for getEntityPropertyList (Reactor E)'
anEntityPropertyList = anEml.getEntityPropertyList( 'Reactor:/CELL/CYTOPLASM:E' )
print anEntityPropertyList, '\n'

## 
print '## TEST for getEntityProperty (Substance S, Quantity)'
anEntityProperty = anEml.getEntityProperty( 'Substance:/CELL/CYTOPLASM:S', 'Quantity' )
print anEntityProperty, '\n'

## 
print '## TEST for getEntityProperty (Reactant E, Reactant)'
anEntityProperty = anEml.getEntityProperty( 'Reactor:/CELL/CYTOPLASM:E', 'Reactant' )
print anEntityProperty, '\n'


