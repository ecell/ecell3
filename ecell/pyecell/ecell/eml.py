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
        self.__theDocument = minidom.parse( aFile )



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

    def __init__( self, aFile ):
        """read EML file and make Document Object"""
        self.__theFile = aFile
        self.__theDocument = minidom.parse( self.__theFile )


    def parse( self ):
        """parse the DOM to the PreModel Object
        - aPreModel
            * aPropertyPreModelElement -> [aFullPn,aValueList]
            * anEntityPreModelElement  -> [anId,anEntityClass,aName]
        """

        aStepperPreModel        = self.getStepperPreModel()
        aStepperSystemPreModel  = self.getStepperSystemPreModel()
        aPropertyPreModel       = self.getPropertyPreModel()
        anEntityPreModel        = self.getEntityPreModel()


        aPreModel = { 'stepper': aStepperPreModel, \
                      'stepper_system': aStepperSystemPreModel, \
                      'property': aPropertyPreModel, \
                      'entity': anEntityPreModel }



        ## Temporary Path Convert for 3.0.0, refactor!!
        aPathConverter = ConvertPath( self.__theFile )
        aPathConverter.createPathList( 'None' )

        for aTargetEntity in( aPreModel['entity'] ):
            aTargetEntity[0] = aPathConverter.change( aTargetEntity[0] )
        for aTargetStepperSystem in( aPreModel['stepper_system'] ):
            aTargetStepperSystem[0] = aPathConverter.change( aTargetStepperSystem[0] )
        ##-----------------------------------------------------------------------------

        return aPreModel



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

            anId        = str( aTargetSystem.getAttribute( 'id' ) )
            aSystemName = str( aTargetSystem.getAttribute( 'name' ) )
            aStepper    = str( aTargetSystem.getAttribute( 'stepper' ) )

            anEntityPreModel.append( [anId,'system',aSystemName] )
            

            for aTargetEntity in( aTargetSystem.childNodes ):

                if aTargetEntity.tagName == 'substance' or \
                   aTargetEntity.tagName == 'reactor':

                    anEntityClass = str( aTargetEntity.tagName )
                    aName         = str( aTargetEntity.getAttribute( 'name' ) )
                    
                    anEntityPreModel.append( [ anId, anEntityClass, aName ] )

        return anEntityPreModel



#---------------------------------------------------------"""
class ConvertPath:
    """convert relative path to absolute path"""

    def __init__( self, anEmlFile ):
        """initialize self.__thePedigreeList"""
        self.__theDocument     = minidom.parse( anEmlFile )
        self.__thePedigreeList = []
        

    def change( self, anId ):
        """turn relative path to absolute path"""


        if anId == '/':
            aPath = '/'
            return aPath

        else:
            for aPedigree in( self.__thePedigreeList ):
                i = 0
                for anElement in( aPedigree ):
                    i = i + 1
                    if anElement == anId:
                        aPathList = aPedigree[0:i]
                        aPath     = string.join( aPathList, '/' )
                        aPath     = str( '/' + aPath )

                        return aPath



    def createPathList( self, aChildName ):
        """create absolute path list"""

        if aChildName == 'None':
            self.__thePedigree = []

        self.createFillitationList()
        
        for aFillitation in( self.__theFillitationList ):


            if aFillitation[1] == aChildName:
                self.__thePedigree.append( aFillitation[1] )
                
                if not aFillitation[0] == '/':
                    self.createPathList( aFillitation[0] )
                    
                else:
                    del self.__thePedigree[0]
                    self.__thePedigree.reverse()
                    self.__thePedigreeList.append( self.__thePedigree )
                    
                    self.__thePedigree = [] ## initialization
                    


    def createFillitationList( self ):
        self.__theFillitationList = []

        for aSystem in( self.__theDocument.getElementsByTagName( 'system' ) ):

            aParentPath = aSystem.getAttribute( 'id' )
            
            if len( aSystem.getElementsByTagName( 'subsystem' ) ) > 0:
                for aChild in ( aSystem.childNodes ):
                    if aChild.tagName == 'subsystem':
                        aChildPath  = aChild.getAttribute( 'id' )
                        self.__theFillitationList.append( [ aParentPath, aChildPath ] )

            else:
                aChildPath  = 'None'
                self.__theFillitationList.append( [ aParentPath, aChildPath ] )



                    
    def returnList( self ):
        """return all absolute path lists"""
        return self.__thePedigreeList



#---------------------------------------------------------"""
class PreModelValidator:

    def validate( self, aPreModel ):
        print 'This method is "validate".'

#    def changeSystemPath( self ):
    

