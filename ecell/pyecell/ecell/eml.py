"""
This is emllib for EML
"""
__author__     = 'suzuki'
__email__      = 'suzuki@e-cell.org'
__startDate__  = '020316'
__lastUpdate__ = '020318'

__Memo__ = '\
- anId means fullPN as tuple [020326]\
'

__Todo__ = '\
- path convert will be done in the parsing process in a lump [020412]\
'


#------------------- window line ------------------------------------------------------------------#

from xml.dom import minidom
import string ## for convertPath, ModelInterpreter


#---------------------------------------------------------"""
class Eml:
    """This class uses Level-1 methods."""

    def __init__( self, aFile ):
        """read EML file and make domtree as a member value"""

        aFileObject = open( aFile )
        aFileList   = aFileObject.readlines()
        aStringData = string.join( string.join( filelist, '' ).split( '\n' ), '' )
        self.__theDocument = minidom.parseString( aStringData )


    def save( self, anOutputFile ):
        """save domtree as an XML file"""
        anOutputObject = open( anOutputFile, 'w' )
        anOutputObject.write( self.__theDocument.toxml() )



    def show( self ):
        """print EML document"""
        print self.__theDocument.toxml()



    def getProperty( self, anId ):
        """return values of property, specified with full PN"""
        aFullId = anId[0] + ':' + anId[1] + ':' + anId[2]
        aName = anId[3]

        for aTargetModel in( self.__theDocument.documentElement.childNodes ):
            aPropertyList = self.searchTag( aTargetModel, 'property' )
            aPropertyList = self.searchAttribute( aPropertyList, 'fullid', aFullId )
            aPropertyList = self.searchAttribute( aPropertyList, 'name', aName )

            ## only one element meets the conditions
            if len( aPropertyList ) == 1:
                return aPropertyList[0].toxml() ## temporary value



    def addProperty( self, aTargetModel, anId, aFix, *aValueList ):
        """add new property with values as many as you like"""
        
        aFullId = anId[0] + ':' + anId[1] + ':' + anId[2]
        aName = anId[3]
        
        aPropertyElement = self.makeElement\
                           ( 'property', '', 'fullid', aFullId, 'name', aName, 'fix', aFix )

        ## make element, 'value'
        ##----------------------------------------------
        aNumber = 0          #
        aValueElement = []   # initialization

        for aTargetValue in( aValueList ):
            [ aValue, anUnit, aRange ] = aTargetValue
            aNumberString = str( aNumber )

            aTargetValueElement = self.makeElement\
                                  ( 'value', aValue, 'unit', anUnit, \
                                    'range', aRange, 'number', aNumberString )
            aNumber = aNumber + 1
            aPropertyElement.appendChild( aTargetValueElement )


        ## connect aPropertyElement to theDocument
        ##-------------------------------------------------------------------

        for aTargetNode in( self.__theDocument.documentElement.childNodes ):
            if aTargetNode.tagName == 'model' and\
               aTargetNode.getAttribute( 'origin' ) == aTargetModel:

                aTargetNode.appendChild( aPropertyElement )



    def deleteProperty( self, *anIdList ):
        """delete properties"""

        for anId in( anIdList ):
            aFullId = anId[0] + ':' + anId[1] + ':' + anId[2]
            aName = anId[3]

            numModels = len( self.__theDocument.documentElement.childNodes )
            for aTargetModel in( self.__theDocument.documentElement.childNodes ):
                for aTargetChild in( aTargetModel.childNodes ):

                    if aTargetChild.tagName == 'property' and \
                       aTargetChild.getAttribute( 'fullid' )   == aFullId and \
                       aTargetChild.getAttribute( 'name' ) == aName:
                        
                        aTargetModel.removeChild( aTargetChild )



    def overwriteProperty( self, anId, aFix, *aValueList ):
        """overwrite fix attr of property and data, unit, range of any values"""

        aFullId = anId[0] + ':' + anId[1] + ':' + anId[2]
        aName = anId[3]
        
        for aTargetModel in( self.__theDocument.documentElement.childNodes ):
            for aTargetChild in( aTargetModel.childNodes ):

                if aTargetChild.tagName == 'property' and \
                   aTargetChild.getAttribute( 'fullid' )   == aFullId and \
                   aTargetChild.getAttribute( 'name' ) == aName:
                
                    ## overwrite attr 'fix'
                    if not aFix == 'None':
                        aTargetChild.setAttribute( 'fix', aFix )


                    ## you must reset all old values
                    aNumberOldValues = len( aTargetChild.childNodes )
                    if len( aValueList ) == aNumberOldValues:
                        for i in range( aNumberOldValues ):
                            anOldValue = aTargetChild.childNodes[i]
                        
                            if not aValueList[i] == 'None':
                                [ aValue, anUnit, aRange ] = aValueList[i]

                                if anOldValue.firstChild:
                                    anOldValue.removeChild( anOldValue.firstChild )
                                    
                                if aValue:
                                    aNewData = self.__theDocument.createTextNode( aValue )
                                    anOldValue.appendChild( aNewData )
                            
                                anOldValue.setAttribute( 'unit', anUnit )
                                anOldValue.setAttribute( 'range', aRange )

                    elif not len( aValueList ) == 0:
                        print '\nERROR: Please input values as many as in existence, or nothing.\n'



    def addEntity( self, aTargetModel, anEntityType, anId, aName, *anItems ):
        """add entity, ie substance, reactor, system
        - addEntity( 'substance', Id, Name, SystemId, SystemName )
        - addEntity( 'reactor'  , Id, Name, Class, SystemId, SystemName )
        - addEntity( 'system'   , Id, Name, Stepper )"""

        
        ## add 'system' beneath 'model' element
        ##-----------------------------------------
        if( anEntityType == 'system' ):
            aStepper = anItems[0]
            aSystemElement = self.makeElement\
                             ( 'system', '', 'id', anId, 'name', aName, 'stepper', aStepper )

            for aTargetChild in( self.__theDocument.firstChild.childNodes ):
                if aTargetChild.tagName == 'model' and\
                   aTargetChild.getAttribute( 'origin' ) == aTargetModel:
                    aTargetChild.appendChild( aSystemElement )


        ## add 'substance' or 'reactor' where you select
        ##-----------------------------------------------
        elif( anEntityType == 'substance' or anEntityType == 'reactor' ):

            [ aSystemId, aSystemName ] = [ anItems[-2], anItems[-1] ]


            ## make element for substance or reactor
            if anEntityType == 'substance':
                anEntityElement = self.makeElement( 'substance', '', 'id', anId, 'name', aName )

            elif anEntityType == 'reactor':
                aClass = anItems[0]
                anEntityElement = self.makeElement\
                                  ( 'reactor', '', 'id', anId, 'name', aName, 'class', aClass )

            ## append substance or reactor beneath a target system
            for aTargetModel in( self.__theDocument.documentElement.childNodes ):
                for aTargetChild in( aTargetModel.childNodes ):
                    if aTargetChild.tagName == 'system' and \
                       aTargetChild.getAttribute( 'id' )   == aSystemId and \
                       aTargetChild.getAttribute( 'name' ) == aSystemName:

                        aTargetChild.appendChild( anEntityElement )



    def deleteEntity( self, anEntityType, anId, aName, *aSystemInfo  ):
        """delete an entity"""

        if( anEntityType == 'system' ):
            aStepper = aSystemInfo[0]

            for aTargetModel in( self.__theDocument.firstChild.childNodes ):

                if aTargetModel.tagName == 'model':
                    ## reduce candidates
                    aSystemList = self.searchTag( aTargetModel, anEntityType )
                    aSystemList = self.searchAttribute( aSystemList, 'id', anId )
                    aSystemList = self.searchAttribute( aSystemList, 'name', aName )
                    aSystemList = self.searchAttribute( aSystemList, 'stepper', aStepper )

                    if len( aSystemList ) == 1:
                        aTargetModel.removeChild( aSystemList[0] )


        elif( anEntityType == 'substance' or anEntityType == 'reactor' ):

            [ aSystemId, aSystemName ] = aSystemInfo

            for aTargetModel in( self.__theDocument.firstChild.childNodes ):

                if aTargetModel.tagName == 'model':
                    ## reduce candidates
                    anEntityList = self.searchTag( aTargetModel, anEntityType )
                    anEntityList = self.searchAttribute( anEntityList, 'id', anId )
                    anEntityList = self.searchAttribute( anEntityList, 'name', aName )

                    aSystemList = self.searchTag( aTargetModel, 'system' )
                    aSystemList = self.searchAttribute( aSystemList, 'id', aSystemId )
                    aSystemList = self.searchAttribute( aSystemList, 'name', aSystemName )

                    if len( anEntityList ) == 1 and len( aSystemList ) == 1:

                        aSystemElement    = aSystemList[0]
                        aNewSystemElement = aSystemList[0]
                        aDeleteElement    = anEntityList[0]

                        aNewSystemElement.removeChild( aDeleteElement )
                        aTargetModel.removeChild( aSystemElement )
                        aTargetModel.appendChild( aNewSystemElement )

                        

            
    def searchTag( self, aMother, aTargetTag ):
        aBranchList = aMother.getElementsByTagName( aTargetTag )
        return aBranchList



    def searchAttribute( self, aTargetTagList, aTargetAttName, aTargetAttValue ):
        aRequiredElementsList = [] ## initialization
        for aTargetTag in( aTargetTagList ):
            if( aTargetAttValue == aTargetTag.getAttribute( aTargetAttName )):
                aRequiredElementsList.append( aTargetTag )        
        return aRequiredElementsList



    def makeElement( self, aTagName, aValue, *anAttrList ):
        """make an element with some attributes as much as you like"""

        ## create new element
        ##---------------------------------------------
        aNewElement = self.__theDocument.createElement( aTagName )
        
        
        ## append attributes
        ##---------------------------------------------
        for i in range( len(anAttrList) ):
            
            if i % 2 == 0:
                anAttrName  = anAttrList[i]
                anAttrValue = anAttrList[ i + 1 ]                
                aNewElement.setAttribute( anAttrName, anAttrValue )

        ## insert data
        ##---------------------------------------------
        if aValue:
            aNewData = self.__theDocument.createTextNode( aValue )
            aNewElement.appendChild( aNewData )

            
        ## return
        ##---------------------------------------------
        return aNewElement





#---------------------------------------------------------"""
class Model:
    """This class uses Level-2 methods."""


    def integrate( self, anOutputFile, *aFiles ):
        """read multi EML files and integrate model elements with origin information"""

        anOutputFile = open( anOutputFile, 'w' )
        anOutputFile.write( '<eml>' )
        
        for aTargetFile in( aFiles ):
            aTargetDocument = minidom.parse( aTargetFile )
            aModelList = aTargetDocument.documentElement.getElementsByTagName( 'model' )

            for aTargetModel in( aModelList ):
                aTargetModel.setAttribute( 'origin', aTargetFile )
                anOutputFile.write( aTargetModel.toxml() )
                
        anOutputFile.write( '</eml>' )



#---------------------------------------------------------"""
class EmlParser:
    """This class parses EML file to PreModel Object."""


    def __init__( self, *aFileObjectList ):
        """read EML file and make Document Object"""


        self.__theDocumentList = []
        for aTargetFileObject in( aFileObjectList ):

            try:
                aFileName = aTargetFileObject.name
            except AttributeError:                      ## with StringData
                aFileName = 'StringData'                ## error will occur!! why!?


            aFileList   = aTargetFileObject.readlines()
            aStringData = string.join( string.join( aFileList, '' ).split( '\n' ), '' )

            aDocument = minidom.parseString( aStringData )

            self.__theDocumentList.append( ( aDocument, aFileName ) )




    def parse( self ):


        self.__thePedigree     = {}
        self.__thePropertyList = []
        self.__theSystemList   = []
        self.__theStepperList  = []

        for aTargetDocument in( self.__theDocumentList ):

            ### get path information ###
            self.getSystemPropertyList( aTargetDocument[0].documentElement )

            ### add Origin to each elements ###
            anOrigin = aTargetDocument[1]
            self.markOrigin( aTargetDocument[0].documentElement, anOrigin )
            
            ### connect DOM tree (Property) ###
            #@#I haven't allowed for metadata yet! @020421
            self.getPropertyList( aTargetDocument[0].documentElement )

            ### connect DOM tree (System) ###
            #@#I haven't allowed for metadata yet! @020421
            self.getSystemList( aTargetDocument[0].documentElement )

            ### connect DOM tree (Stepper) ###
            self.getStepperList( aTargetDocument[0].documentElement )


        aDomList = { 'property': self.__thePropertyList, \
                     'system'  : self.__theSystemList,   \
                     'stepper' : self.__theStepperList   \
                   }


        aRootString = '<eml><model></model></eml>'
        self.__theDocument = minidom.parseString( aRootString )

        for aTargetType in( 'property', 'system', 'stepper' ):
            for aTargetElement in( aDomList[ aTargetType ] ):
                self.__theDocument.documentElement.appendChild( aTargetElement )


        ###                                                 ###
        ### if PreModel is aborted, please change following ###
        ###                                                 ###


        aStepperPreModel        = self.getStepperPreModel()
        aStepperSystemPreModel  = self.getStepperSystemPreModel()
        aPropertyPreModel       = self.getPropertyPreModel()
        anEntityPreModel        = self.getEntityPreModel()

	anStepperSystemPreModel = self.convertStepperSystemPath( aStepperSystemPreModel )
        anEntityPreModel        = self.convertEntityPath( anEntityPreModel )
	

        aPreModel = { 'stepper'       : aStepperPreModel,       \
                      'stepper_system': aStepperSystemPreModel, \
                      'property'      : aPropertyPreModel,      \
                      'entity'        : anEntityPreModel        \
                    }

        return aPreModel





    def showPreModel( self, aPreModel ):
        for aTargetType in( 'stepper', 'stepper_system', 'property', 'entity' ):
            print aTargetType,':'
            for aTarget in( aPreModel[ aTargetType ] ):
                print aTarget
            print '\n'
                        



##### methods for methods #####
        
    def markOrigin( self, aTargetNode, anOrigin ):
        """for parse method, this works recursively"""

        if len( aTargetNode.childNodes ) > 0:
            for aChild in( aTargetNode.childNodes ):
                try:
                    aChild.setAttribute( 'origin', anOrigin )
                except AttributeError:
                    pass
                self.markOrigin( aChild, anOrigin )



    def removeOrigin( self, aTargetNode ):
        """this works recursively"""
        pass




    def getSystemPropertyList( self, aTargetNode ):
        """for parse method, this works recursively"""

        if len( aTargetNode.childNodes ) > 0:
            for aChild in( aTargetNode.childNodes ):
                try:
                    if aChild.tagName == 'property' and \
                       aChild.getAttribute( 'fullid' ).split( ':' )[0] == 'System':

                        aFullId = aChild.getAttribute( 'fullid' ).split( ':' )
                        aRelativePath = aFullId[2]
                        
                        if not ( aFullId[2] == '/' or aFullId[1] == '/' ):
                            anAbsolutePath = aFullId[1] + '/' + aFullId[2]
                        else:
                            anAbsolutePath = aFullId[1] + aFullId[2]


                        ## system-subsystem overwrite
                        anAbsolutePathList = self.__thePedigree.values()
                        for aCheckingAbsolutePath in( anAbsolutePathList ):
                            if anAbsolutePath == aCheckingAbsolutePath:
                                self.__thePedigree.remove\
                                                  ( self.__thePedigree[ aCheckingAbsolutePath ] )
                                
                        self.__thePedigree[ aRelativePath ] = anAbsolutePath

                        
                except AttributeError:
                    pass
                self.getSystemPropertyList( aChild )




    def getPropertyList( self, aTargetNode ):
        """for parse method, this works recursively"""

        if len( aTargetNode.childNodes ) > 0:
            for aTargetChild in( aTargetNode.childNodes ):
                try:
                    if aTargetChild.tagName == 'property':

                        aTargetFullid = aTargetChild.getAttribute( 'fullid' )
                        aTargetName   = aTargetChild.getAttribute( 'name' )


                        ## remove old element for Property list overwrite
                        for aCheckingProperty in( self.__thePropertyList ):

                            aCheckingFullid = aCheckingProperty.getAttribute( 'fullid' )
                            aCheckingName   = aCheckingProperty.getAttribute( 'name' )

                            if \
                               aCheckingProperty.getAttribute( 'name' ) == 'Reactant' and \
                               self.compareProperty( aCheckingProperty, aTargetChild ) == 1:

                                ## cannot specify aReactantProperty with fullid and name
                                ## specify with all value data and attributes

                                self.__thePropertyList.remove( aCheckingProperty )

                            elif \
                               not aCheckingProperty.getAttribute( 'name' ) == 'Reactant' and \
                               aTargetFullid == aCheckingFullid and \
                               aTargetName   == aCheckingName:

                                self.__thePropertyList.remove( aCheckingProperty )


                        self.__thePropertyList.append( aTargetChild )

                except AttributeError:
                    pass
                self.getPropertyList( aTargetChild )




    def getSystemList( self, aTargetNode ):
        """for parse method, this works recursively"""
        
        for aTargetChild in( aTargetNode.childNodes ):
            try:
                if aTargetChild.tagName == 'system':
                    aTargetId      = aTargetChild.getAttribute( 'id' )
                    aTargetStepper = aTargetChild.getAttribute( 'stepper' )

                    ## remove old element for System list overwrite
                    for aCheckingSystem in( self.__theSystemList ):
                        aCheckingId      = aCheckingSystem.getAttribute( 'id' )
                        aCheckingStepper = aCheckingSystem.getAttribute( 'stepper' )

                        #if self.compareSystem( aCheckingSystem, aTargetChild ) == 1:
                        if aTargetId      == aCheckingId and \
                           aTargetStepper == aCheckingStepper:
                            self.__theSystemList.remove( aCheckingSystem )
                            
                    self.__theSystemList.append( aTargetChild )

            except AttributeError:
                pass

            self.getSystemList( aTargetChild )



    def getStepperList( self, aTargetNode ):
        """for parse method, this works recursively"""
        
        for aTargetChild in( aTargetNode.childNodes ):
            try:
                if aTargetChild.tagName == 'stepperlist':
                    aTargetId      = aTargetChild.getAttribute( 'id' )
                    aTargetStepper = aTargetChild.getAttribute( 'stepper' )

                    ## remove old element for System list overwrite
                    for aCheckingStepper in( self.__theStepperList ):
                        anId     = aCheckingStepper.getAttribute( 'id' )
                        aStepper = aCheckingStepper.getAttribute( 'stepper' )

                        if aTargetId      == anId and \
                           aTargetStepper == aStepper:
                            self.__theStepperList.remove( aCheckingStepper )

                    self.__theStepperList.append( aTargetChild )

            except AttributeError:
                pass

            self.getStepperList( aTargetChild )




    def compareProperty( self, anElementA, anElementB ):

        def setOrigin( aTargetElement, anOrigin ):
            aTargetElement.setAttribute( 'origin', anOrigin )
            if len( aTargetElement.childNodes ) > 0:
                for aTargetChild in( aTargetElement.childNodes ):
                    aTargetChild.setAttribute( 'origin', anOrigin )

        def removeOrigin( aTargetElement ):
            setOrigin( aTargetElement, '' )


        anOriginA = anElementA.getAttribute( 'origin' )
        anOriginB = anElementB.getAttribute( 'origin' )

        removeOrigin( anElementA )
        removeOrigin( anElementB )

        aCompareResult =  self.compareDomElement( anElementA, anElementB )

        setOrigin( anElementA, anOriginA )
        setOrigin( anElementB, anOriginB )

        return aCompareResult

    
### very dangerous because this don't allow for metadata without any option   ###
### I recommend not to use this comparint (suzuki@020421)                     ###


    def compareDomElement( self, anElementA, anElementB ):
        aStringA = anElementA.toxml()
        aStringB = anElementB.toxml()

        #print aStringA  ##
        #print aStringB  ##DebugMessage

        if aStringA == aStringB:
            return 1
        else:
            return 0




    def getStepperPreModel( self ):
        """for parse"""
        aStepperlistList = self.__theDocument.getElementsByTagName( 'stepperlist' )
        aStepperPreModel = []
        
        for aTargetStepper in( aStepperlistList ):
            aFullId = str( aTargetStepper.getAttribute( 'id' ) )
            aClass  = str( aTargetStepper.getAttribute( 'class' ) )
            
            aStepperPreModel.append( [ aClass, aFullId ] )

        return aStepperPreModel
        ## [ stepperClass, stepperId ]



    def getStepperSystemPreModel( self ):
        """for parse"""
        aSystemList      = self.__theDocument.getElementsByTagName( 'system' )
        aStepperSystemPreModel = []

        for aTargetSystem in( aSystemList ):
            aFullId  = str( aTargetSystem.getAttribute( 'id' ) )
            aStepper = str( aTargetSystem.getAttribute( 'stepper' ) )

            aStepperSystemPreModel.append( [ aFullId, aStepper] )

        return aStepperSystemPreModel
        ## [ systemFullPn, stepperId ]




    def getPropertyPreModel( self ):
        """for parse"""

        aPropertyList = self.__theDocument.getElementsByTagName( 'property' )
        aPropertyPreModel = []
        
        for aTargetProperty in( aPropertyList ):

            ## fix information
            aFix    = aTargetProperty.getAttribute( 'fix' )            


            ## FullPN information
            aFullid = aTargetProperty.getAttribute( 'fullid' )
            aName   = aTargetProperty.getAttribute( 'name' )
            aFullPn = aFullid + ':' + aName


            ## values information
            aValueList = []
            for aTargetChild in( aTargetProperty.childNodes ):

                if aTargetChild.tagName == 'value':
                    aValue = aTargetChild.firstChild.data
                    aRange = aTargetChild.getAttribute( 'range' )

                    if aValue == 'unknown':
                        aValue = aRange

                ## append to valueList
                aValue = str( aValue ) ## for Unicode
                aValueList.append( aValue )

            aFullPn    = str( aFullPn ) ## for Unicode
            aPropertyPreModel.append( [ aFullPn, aValueList ] )

        return aPropertyPreModel




    def getEntityPreModel( self ):
        """for parse"""

        aSystemList = self.__theDocument.getElementsByTagName( 'system' )
        anEntityPreModel = []
    
        for aTargetSystem in( aSystemList ):

            aPath     = str( aTargetSystem.getAttribute( 'id' ) )
            aPathList = aPath.split( '/' )

            aSystemId = aPathList[-1]
            if not aSystemId:
                aSystemId = '/'

            aSystemPath = aPath
            aSystemName = str( aTargetSystem.getAttribute( 'name' ) )
            aStepper    = str( aTargetSystem.getAttribute( 'stepper' ) )
            
            anEntityPreModel.append( [ 'System', aSystemPath, aSystemId, aSystemName ] )


            for aTargetEntity in( aTargetSystem.childNodes ):

                if aTargetEntity.tagName == 'substance' or \
                   aTargetEntity.tagName == 'reactor':

                    
                    aPath         = aSystemPath
                    anEntityClass = string.capitalize( str( aTargetEntity.tagName ) )

                    if anEntityClass == 'Reactor':
                        anEntityClass = str( aTargetEntity.getAttribute( 'class' ) )

                    aName         = str( aTargetEntity.getAttribute( 'name' ) )
                    anId          = str( aTargetEntity.getAttribute( 'id' ) )

                    anEntityPreModel.append( [ anEntityClass, aPath, anId, aName ] )

        return anEntityPreModel



    def convertEntityPath( self, anEntityPreModel ):

        for aTargetEntity in( anEntityPreModel ):
            aRelativePath  = aTargetEntity[1]
            try:
                anAbsolutePath = self.__thePedigree[ aRelativePath ]
            except KeyError:
                anAbsolutePath = 'AbsolutePathError'

            aTargetEntity[1] = str( anAbsolutePath )
            
        return anEntityPreModel




    def convertStepperSystemPath( self, aStepperSystemPreModel ):

        for aTargetSS in( aStepperSystemPreModel ):
            aRelativePath  = aTargetSS[0]
            try:
                anAbsolutePath = self.__thePedigree[ aRelativePath ]
            except KeyError:
                anAbsolutePath = 'AbsolutePathError'

            aTargetSS[0] = str( anAbsolutePath )
            
        return aStepperSystemPreModel




#---------------------------------------------------------"""
class FileValidator:
    """Level-1 validation for each file"""


    def __init__( self, aFileObject ):
        aFileList   = aFileObject.readlines()
        aStringData = string.join( string.join( aFileList, '' ).split( '\n' ), '' )
        try:
            self.__theDocument = minidom.parseString( aStringData )
        except:
            print 'FILE VALIDATION ERROR: cannot parse\n\n'

        else:
            self.checkProperty()
            self.checkStepper()
            self.checkSystem()




###----------------------------------------------------------------------------------
### Porperty Validation
###__________________________________________________________________________________

    def checkProperty( self ):
        aModelNode = self.__theDocument.childNodes[0].childNodes[0]

        aPropertyList = []
        for aTargetChild in aModelNode.childNodes:
            if aTargetChild.tagName == 'property':
                aPropertyList.append( aTargetChild )


        for aTargetProperty in aPropertyList:
            aFullId = aTargetProperty.getAttribute( 'fullid' )
            aName   = aTargetProperty.getAttribute( 'name' )
            
            aValueList = []
            for aTargetChild in aTargetProperty.childNodes:
                if aTargetChild.tagName == 'value':
                    aValueList.append( aTargetChild )
                elif aTargetChild.tagName == 'metadata':
                    pass
                else:
                    self.showError( aTargetChild.tagName, 'There is an improper element.')



            self.checkAttributeExistence( aTargetProperty.toxml(), 'fullid', aFullId )
            self.checkAttributeExistence( aTargetProperty.toxml(), 'name', aName )
            self.checkFullId( aFullId )
            self.checkValueList( aValueList )
                
            if len( aValueList ) == 0:
                self.showError( aTargetProperty.toxml(), 'Property element needs value elements.' )
                


###----------------------------------------------------------------------------------
### Stepper Validation
###__________________________________________________________________________________

    def checkStepper( self ):
        aModelNode = self.__theDocument.childNodes[0].childNodes[0]
        
        aStepperList = []
        for aTargetChild in aModelNode.childNodes:
            if aTargetChild.tagName == 'stepperlist':
                aStepperList.append( aTargetChild )

        for aTargetStepper in aStepperList:
            aTargetClass = aTargetStepper.getAttribute( 'class' )
            aTargetId    = aTargetStepper.getAttribute( 'id' )
            
            self.checkAttributeExistence( aTargetStepper.toxml(), 'class', aTargetClass )
            self.checkAttributeExistence( aTargetStepper.toxml(), 'id', aTargetId )
            



###----------------------------------------------------------------------------------
### System Validation
###__________________________________________________________________________________

    def checkSystem( self ):
        aModelNode = self.__theDocument.childNodes[0].childNodes[0]
        
        aSystemList = []
        for aTargetChild in aModelNode.childNodes:
            if aTargetChild.tagName == 'system':
                aSystemList.append( aTargetChild )


        aSubsystemList = []
        for aTargetSystem in aSystemList:
            for aTargetChild in aTargetSystem.childNodes:
                if aTargetChild.tagName == 'subsystem':
                    aSubsystemList.append( aTargetChild )
                    

        ## check existence of attribute
        for aTargetSystem in aSystemList:
            aTargetStepper = aTargetSystem.getAttribute( 'stepper' )
            aTargetId      = aTargetSystem.getAttribute( 'id' )
            aTargetName    = aTargetSystem.getAttribute( 'name' )
            
            self.checkAttributeExistence( aTargetSystem.toxml(), 'stepper', aTargetStepper )
            self.checkAttributeExistence( aTargetSystem.toxml(), 'id', aTargetId )
            self.checkAttributeExistence( aTargetSystem.toxml(), 'name', aTargetName )

            

        ## Parents must know their children. ( check mother > child )
        for aTargetSubsystem in aSubsystemList:
            aSubsystemExistence = 0
            aTargetSubsystemId = aTargetSubsystem.getAttribute( 'id' )

            for aTargetSystem in aSystemList:
                aTargetSystemId = aTargetSystem.getAttribute( 'id' )

                if aTargetSubsystemId == aTargetSystemId:
                    aSubsystemExistence = 1

            if aSubsystemExistence == 0:
                self.showError( aTargetSubsystem.toxml(), "This mother don't know her child." )
            

        ## Children must know their parents.( check child > mother )
        for aTargetSystem in aSystemList:
            aSystemExistence = 0
            aTargetSystemId = aTargetSystem.getAttribute( 'id' )

            for aTargetSubsystem in aSubsystemList:
                aTargetSubsystemId = aTargetSubsystem.getAttribute( 'id' )
                
                if aTargetSystemId == aTargetSubsystemId:
                    aSystemExistence = 1

            if aSystemExistence == 0 and \
               aTargetSystemId != '/':
                self.showError( aTargetSystem.toxml(), "This child don't know his mother." )
                


        ## check attributes of Subsystem, Substance and Reactor
        for aTargetSystem in aSystemList:
            for aTargetChild in aTargetSystem.childNodes:
                aTargetTagName = aTargetChild.tagName

                if aTargetTagName   == 'subsystem':
                    self.checkSubsystem( aTargetChild )

                elif aTargetTagName == 'substance':
                    self.checkSubstance( aTargetChild )

                elif aTargetTagName == 'reactor':
                    self.checkReactor( aTargetChild )



    #----------------------------------------------------------------------------------
    # For Porperty Validation
    #__________________________________________________________________________________

    def checkFullId( self, aFullId ):
        aFullIdList = aFullId.split( ':' )


        if len( aFullIdList ) != 3:
            self.showError( aFullId, 'FullId must have 3 elements separated by semicolon.' )


        if aFullIdList[0] != 'System'    and \
           aFullIdList[0] != 'Substance' and \
           aFullIdList[0] != 'Reactor':
            self.showError\
             ( aFullIdList[0], 'First element of fullId must be System, Substance or Reactor.' )



    def checkValueList( self, aValueList ):
        aNumValueList = len( aValueList )

        aCounter = 0
        for aTargetValue in aValueList:
            aTargetNumber = aTargetValue.getAttribute( 'number' )
            aTargetNumber = int( aTargetNumber )
            
            if aTargetNumber != aCounter:
                self.showError( aTargetValue.toxml(), "The 'number' must be sequential number." )
            aCounter = aCounter + 1


        for aTargetValue in aValueList:
            if len( aTargetValue.childNodes ) == 0 and \
               aTargetValue.getAttribute( 'range' ) == '':
                self.showError( aTargetValue.toxml(), 'Value element must have value or range.' )
                
                


    #----------------------------------------------------------------------------------
    # For System Validation
    #__________________________________________________________________________________

    def checkSubsystem( self, aTargetSubsystem ):
        aTargetId = aTargetSubsystem.getAttribute( 'id' )
        self.checkAttributeExistence( aTargetSubsystem.toxml(), 'id', aTargetId )
        

    def checkSubstance( self, aTargetSubstance ):
        aTargetId   = aTargetSubstance.getAttribute( 'id' )
        aTargetName = aTargetSubstance.getAttribute( 'name' )
        self.checkAttributeExistence( aTargetSubstance.toxml(), 'id', aTargetId )
        self.checkAttributeExistence( aTargetSubstance.toxml(), 'name', aTargetName )


    def checkReactor( self, aTargetReactor ):
        aTargetClass = aTargetReactor.getAttribute( 'class' )
        aTargetId    = aTargetReactor.getAttribute( 'id' )
        aTargetName  = aTargetReactor.getAttribute( 'name' )
        self.checkAttributeExistence( aTargetReactor.toxml(), 'class', aTargetClass )
        self.checkAttributeExistence( aTargetReactor.toxml(), 'id', aTargetId )
        self.checkAttributeExistence( aTargetReactor.toxml(), 'name', aTargetName )

                


    #----------------------------------------------------------------------------------
    # General Methods for Validation
    #__________________________________________________________________________________
                
    def checkAttributeExistence( self, aTargetElement, aTargetAttributeName, aTargetAttribute ):
        if aTargetAttribute == '':
            aMessage = "This element needs '" + aTargetAttributeName + "' attribute."
            self.showError( aTargetElement, aMessage )



    def showError( self, anErrorPart, aMessage ):
        print 'FILE VALIDATION ERROR:',aMessage,'->',anErrorPart
        
