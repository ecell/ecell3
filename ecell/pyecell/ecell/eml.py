"""
This is emllib for EML
"""
__author__     = 'suzuki'
__email__      = 'suzuki@sfc.keio.ac.jp'
__startDate__  = '020316'
__lastUpdate__ = '020801'

__Memo__ = '\
'

__Todo__ = '\
- How to use multi stepper? Only one stepper can be usable. [020708]\
+ How to overwrite Property? Think about values. [020701] finished'


#------------------- window line ------------------------------------------------------------------#

from xml.dom import minidom
import string


#---------------------------------------------------------"""
class Eml:


    def __init__( self, aFileObject ):
        """read EML file and make domtree"""

        #aFileObject = open( aFile )
        aFileList   = aFileObject.readlines()
        aStringData = string.join( string.join( aFileList, '' ).split( '\n' ), '' )
        self.__theDocument = minidom.parseString( aStringData )



    def asString( self ):
        """return domtree as string"""

        return self.__theDocument.toxml()



    def save( self, anOutputFile ):
        """save domtree as an EML file"""
        
        anEmlString = self.asString()
        
        anOutputFileObject = open( anOutputFile, 'w' )
        anOutputFileObject.write( anEmlString )



    ##---------------------------------------------
    ## Methods for Stepper
    ##---------------------------------------------

    def createStepperlist( self ):
        """create a stepperlist"""

        anElement = self.createElement( 'stepperlist' )
        self.__theDocument.documentElement.\
                           childNodes.append( anElement )



    def createStepper( self, aClass, anId, aValueList ):
        """create a stepper"""
        aStepperElement = self.createElement( 'stepper' )
        aStepperElement.setAttribute( 'class', aClass )
        aStepperElement.setAttribute( 'id', anId )

        aValueId = 0
        for aValue in aValueList:
            aValueElement = self.createElement( 'value' )
            aValueElement.setAttribute( 'number', str( aValueId ) )
            aValueData = self.__theDocument.createTextNode( aValue )
            aValueElement.appendChild( aValueData )
            aStepperElement.appendChild( aValueElement )
            
            aValueId = aValueId + 1


        for aTargetNode in self.__theDocument.documentElement.childNodes:
            if aTargetNode.tagName == 'stepperlist':
                aTargetNode.childNodes.append( aStepperElement )

        

    def deleteStepperlist( self ):
        """delete a stepperlist"""
        for anElement in self.__theDocument.firstChild.childNodes:
            if anElement.tagName == 'stepperlist':
                self.__theDocument.firstChild.removeChild( anElement )


    
    def deleteStepper( self, aClass, anId ):
        """delete a stepper"""
        for anElement in self.__theDocument.firstChild.childNodes:
            if anElement.tagName == 'stepperlist':
                aStepperlistNode = anElement
                for aChildElement in aStepperlistNode.childNodes:
                    if aChildElement.tagName == 'stepper' and \
                       aChildElement.getAttribute( 'class' ) == aClass and \
                       aChildElement.getAttribute( 'id' )    == anId:

                        aStepperlistNode.removeChild( aChildElement )
        

    



    ##---------------------------------------------
    ## Methods for Entity
    ##---------------------------------------------

    def createEntity( self, anEntityType, aClass, aFullId, aName ):
        """create an entity: system, substance and reactor"""


        #if self.checkEntityExistence( aFullId )[0]:
        #    print 'already exists'
        #    return 'error'
        #    ## I want to stop this process here!!!

        anEntityElement = self.createElement( string.lower( anEntityType ) )
        anEntityElement.setAttribute( 'class', aClass )
        anEntityElement.setAttribute( 'name' , aName )


        if( anEntityType == 'System' ):

            anId = self.convertSystemFullId2SystemId( aFullId )
            anEntityElement.setAttribute( 'id', anId )
            self.__theDocument.documentElement.appendChild( anEntityElement )

        elif( anEntityType == 'Substance' or anEntityType == 'Reactor' ):

            anId = aFullId.split( ':' )[2]
            anEntityElement.setAttribute( 'id', anId )

            aTargetFullPath = aFullId.split( ':' )[1]
            for aTargetNode in self.__theDocument.documentElement.childNodes:

                if aTargetNode.tagName == 'system':
                    aTargetSystem = aTargetNode

                    aSystemFullPath = self.asSystemPath( aTargetSystem )

                
                    if aTargetFullPath == aSystemFullPath:
                        aTargetSystem.appendChild( anEntityElement )





                        
    def deleteEntity( self, aFullId ):
        """delete an entity"""

        aTargetEntity = self.asEntityInfo( aFullId )

        if aTargetEntity[ 'Type' ] == 'System':
            for anElement in self.__theDocument.firstChild.childNodes:
                if anElement.getAttribute( 'id' ) == aFullId:
                    self.__theDocument.firstChild.removeChild( anElement )

        else:
            for anElement in self.__theDocument.firstChild.childNodes:
                if anElement.tagName == 'system':
                    if self.asSystemPath( anElement ) == aTargetEntity[ 'Path' ]:

                        for aChild in anElement.childNodes:
                            if aChild.tagName == string.lower( aTargetEntity[ 'Type' ] ) and \
                               aChild.getAttribute( 'id' ) == aTargetEntity[ 'Id' ]:

                                anElement.removeChild( aChild )
                        


    def checkEntityExistence( self, aFullId ):

        aTargetEntity = self.asEntityInfo( aFullId )

        anExistence = 0
        anExistingSystem = 'None'

        aSystemList = self.__theDocument.getElementsByTagName( 'system' )
        for aSystem in aSystemList:

            ## for System
            if aTargetEntity[ 'Type' ] == 'System':
                aSystemPath = aSystem.getAttribute( 'id' ).split( ':' )[1]
                if aSystemPath == aTargetEntity[ 'Path' ]:
                    anExistence = 1
                    anExistingSystem = aSystem

            ## for Substance or Reactor
            else:
                aSystemPath = self.asSystemPath( aSystem )

                if aSystemPath == aTargetEntity[ 'Path' ]:
                    for anElement in aSystem.childNodes:

                        if anElement.tagName == string.lower( aTargetEntity[ 'Type' ] ) and \
                           anElement.getAttribute( 'id' ) == aTargetEntity[ 'Id' ]:

                            anExistence = 1
                            anExistingSystem = aSystem
                            
        return ( anExistence, anExistingSystem )





    ##---------------------------------------------
    ## Methods for Property
    ##---------------------------------------------
    
    def setEntityProperty( self, aFullId, aName, aValueList ):

        #aPropertyExistence = self.checkPropertyExistence( aFullId )
        #if self.checkPropertyExistence( aFullId ):
        #    print 'already exists'
        #    return 'error'
        aPropertyExistence = 0 ## Temporary [020730]

        aTargetEntity = self.asEntityInfo( aFullId )

        if aPropertyExistence == 0:

            aPropertyElement = self.createPropertyElement( aName, aValueList )

            for aTargetNode in self.__theDocument.firstChild.childNodes:
                if aTargetNode.tagName == 'system':
                    aTargetSystem = aTargetNode

                    aTargetSystemId     = aTargetSystem.getAttribute( 'id' )
                    aTargetSystemFullId = self.convertSystemId2SystemFullId( aTargetSystemId )
                    aTargetSystemPath   = self.asSystemPath( aTargetSystem )


                    ## for System
                    if aTargetEntity[ 'Type' ] == 'System' and \
                           aTargetSystemFullId == aFullId:
                        
                        aTargetSystem.appendChild( aPropertyElement )


                    ## for Substance or Reactor
                    elif aTargetSystemPath == aTargetEntity[ 'Path' ]:

                        for anElement in aTargetSystem.childNodes:
                            if anElement.tagName == string.lower( aTargetEntity[ 'Type' ] ) and \
                               anElement.getAttribute( 'id' ) == aTargetEntity[ 'Id' ]:

                                anElement.appendChild( aPropertyElement )





    def checkPropertyExistence( self, aFullId ):
        """This class should be independent of checkEntityExistence method."""

        aTargetEntity = self.asEntityInfo( aFullId )
        aPropertyExistence = 0


        aSystemList = self.__theDocument.getElementsByTagName( 'system' )

        if aTargetEntity[ 'Type' ] == 'System':
            for aSystem in aSystemList:
                if aSystem.getAttribute( 'id' ).split( ':' )[2] == aTargetEntity[ 'Id' ]:

                    for anElement in aSystem.childNodes:
                        if anElement.tagName == 'property':
                            aPropertyExistence = 1

        ## for Substance (Think about Reactor!!)
        elif aTargetEntity[ 'Type' ] == 'Substance':
            for aSystem in aSystemList:
                aSystemPath = self.asSystemPath( aSystem )
                
                if aSystemPath == aTargetEntity[ 'Path' ]:
                    for anElement in aSystem.childNodes:
                    
                        if ( anElement.tagName == 'substance' or \
                             anElement.tagName == 'reactor' ) and \
                             anElement.getAttribute( 'id' ) == aTargetEntity[ 'Id' ]:

                            for anElement in anElement.childNodes:
                                if anElement.tagName == 'property':
                            
                                    aPropertyExistence = 1

        return aPropertyExistence




    def deleteProperty( self, aFullId, aName, aValueList ):
        # deleteCheck -> system, ok!


        aPropertyElement = self.createPropertyElement( aName, aValueList )

        for aSystem in self.__theDocument.getElementsByTagName( 'system' ):
            for anElementFirst in aSystem.childNodes:
                if anElementFirst.toxml() == aPropertyElement.toxml():
                    ## against property of system
                    aSystem.removeChild( anElementFirst )

                if len( anElementFirst.childNodes ) > 0 :
                    for anElementSecond in anElementFirst.childNodes:
                        if anElementSecond.toxml() == aPropertyElement.toxml():
                            ## against property of substance or reactor
                            anElementFirst.removeChild( anElementSecond )
        




    ##---------------------------------------------
    ## Methods for Read
    ##---------------------------------------------


    def getStepperList( self ):

        aStepperList = []
        for aStepperElement in self.getElementsByTagName( 'stepper' ):

            aStepper = {}
            aStepper[ 'Class' ] = str( aStepperElement.getAttribute( 'class' ) )
            aStepper[ 'Id' ]    = str( aStepperElement.getAttribute( 'id' ) )
            aStepper[ 'ValueList' ] = []

            for aChildElement in aStepperElement.childNodes:

                if aChildElement.tagName == 'value':

                    aStepper[ 'ValueList' ].\
                              append( str( aChildElement.firstChild.data ) )
            
            aStepperList.append( aStepper )

        return aStepperList



    def getEntityList( self ):

        aSystemEntityList             = self.getSystemEntityList()
        aSubstanceOrReactorEntityList = self.getSubstanceOrReactorEntityList()
        #aReactorEntityList           = self.getReactorEntityList()
        anEntityList                  = aSystemEntityList + aSubstanceOrReactorEntityList

        return anEntityList


    def getEntityPropertyList( self ):

        aSystemPropertyList  = self.getSystemPropertyList()
        aSubstanceOrReactorPropertyList = self.getSubstanceOrReactorPropertyList()

        aPropertyList = aSystemPropertyList + aSubstanceOrReactorPropertyList

        return aPropertyList        



    def getStepperPropertyList( self ):

        aStepperPropertyList = []


        #aStepperList = []
        #aStepperlistElement = self.__theDocument.getElementsByTagName( 'stepperlist' )[0]
        #for aStepper in aStepperlistElement.childNodes:
        #    if aStepper.tagName == 'stepper':
        #        aStepperId    = str( aStepper.getAttribute( 'id' ) )
        #        aStepperList.append( aStepperId )


        for aSystemElement in self.__theDocument.getElementsByTagName( 'system' ):
            aStepperProperty = {}
            aStepperProperty[ 'FullPn' ] = str( 'System:' + self.asPathToSystem( aSystemElement.getAttribute( 'id' ) ) + ':StepperID' )


            ## Initialization
            aStepperId = ''
            aStepperProperty[ 'StepperId' ] = ''
            
            for aChildElementOfSystemElement in aSystemElement.childNodes:

                if aChildElementOfSystemElement.tagName == 'property' and \
                   aChildElementOfSystemElement.getAttribute( 'name' ) == 'StepperID':

                    aStepperIdPropertyElement =  aChildElementOfSystemElement

                    for aChildElementOfStepperIdProperty in aStepperIdPropertyElement.childNodes:
                        if aChildElementOfStepperIdProperty.tagName == 'value':
                            aStepperId = aChildElementOfStepperIdProperty.firstChild.toxml()

            aStepperProperty[ 'StepperId' ] = [ aStepperId ]            
            aStepperPropertyList.append( aStepperProperty )

        return aStepperPropertyList






        
    # Method for Method
    #------------------------------------------------------------------------------------
    def getSystemPropertyList( self ):
        aPropertyList = []
        for aSystemElement in self.__theDocument.getElementsByTagName( 'system' ):

            aSystemPath = self.asSystemPath( aSystemElement )

            for aChildElement in aSystemElement.childNodes:

                ## Property for System
                if aChildElement.tagName == 'property':
                    aPropertyElement = aChildElement
                    aProperty = {}

                    aPropertyName = aPropertyElement.getAttribute( 'name' )
                    aProperty[ 'FullPn' ] = str( 'System:' + self.asPathToSystem( aSystemElement.getAttribute( 'id' ) ) + ':' + aPropertyName )

                    aValueTextList = []
                    for aValueCandidate in aPropertyElement.childNodes:
                        if aValueCandidate.tagName == 'value':
                            aValueText = str( aValueCandidate.firstChild.toxml() )
                            aValueTextList.append( aValueText )

                    aProperty[ 'ValueList' ] =  aValueTextList
                    aPropertyList.append( aProperty )
                    
        return aPropertyList


    def getSubstanceOrReactorPropertyList( self ):
        aPropertyList = []
        for aSystemElement in self.__theDocument.getElementsByTagName( 'system' ):

            aSystemPath = self.asSystemPath( aSystemElement )
            for aChildElement in aSystemElement.childNodes:
                
                ## Property for Substance or Reactor
                if aChildElement.tagName == 'substance' or \
                   aChildElement.tagName == 'reactor':

                    anEntityType = string.capwords( aChildElement.tagName )
                    aSubstanceElement = aChildElement


                    for aPropertyElement in aSubstanceElement.childNodes:
                        if aPropertyElement.tagName == 'property':

                            aProperty = {}

                            aPropertyName = aPropertyElement.getAttribute( 'name' )
                            aProperty[ 'FullPn' ] = str( anEntityType + ':' + aSystemPath + ':' + \
                                                         aChildElement.getAttribute( 'id' ) + ':' + \
                                                         aPropertyName )

                            aProperty[ 'ValueList' ] = []
                            for aValueCandidate in aPropertyElement.childNodes:
                                if aValueCandidate.tagName == 'value':
                                    aValueText = str( aValueCandidate.firstChild.toxml() )
                                    aProperty[ 'ValueList' ].append( aValueText )

                            aPropertyList.append( aProperty )

        return aPropertyList





    def getSystemEntityList( self ):

        aSystemEntityList = []
        for aSystemElement in self.__theDocument.getElementsByTagName( 'system' ):
            aSystem = {}

            aSystem[ 'Type' ]   = str( aSystemElement.getAttribute( 'class' ) )
            aSystem[ 'FullId' ] = str( 'System:' \
                                       + self.asPathToSystem( aSystemElement.getAttribute( 'id' ) ) )
            aSystem[ 'Name' ]   = str( aSystemElement.getAttribute( 'name' ) )
            
            if not aSystem[ 'FullId' ] == 'System::/':
                aSystemEntityList.append( aSystem )

        return aSystemEntityList



    def getSubstanceOrReactorEntityList( self ):

        anEntityEntityList = []
        for aSystemElement in self.__theDocument.getElementsByTagName( 'system' ):

            aSystemPath = self.asSystemPath( aSystemElement )

            for aChildElement in aSystemElement.childNodes:

                if aChildElement.tagName == 'substance' or \
                   aChildElement.tagName == 'reactor':

                    anEntity = {}                    
                    anEntity[ 'Type' ] = str( aChildElement.getAttribute( 'class' ) )
                    anEntity[ 'FullId' ] = str( string.capwords( aChildElement.tagName ) + ':' + \
                                                aSystemPath + ':' + \
                                                aChildElement.getAttribute( 'id' ) )
                    anEntity[ 'Name' ]   = str( aChildElement.getAttribute( 'name' ) )
                    
                    anEntityEntityList.append( anEntity )    
        return anEntityEntityList




#    def getReactorEntityList( self ):
#
#        anEntityEntityList = []
#        for aSystemElement in self.__theDocument.getElementsByTagName( 'system' ):
#
#            aSystemPath = self.asSystemPath( aSystemElement )
#
#            for aChildElement in aSystemElement.childNodes:
#
#                if aChildElement.tagName == 'reactor':
#
#                    anEntity = {}
#                    anEntity[ 'Type' ]   = str( aChildElement.getAttribute( 'class' ) )
#                    anEntity[ 'FullId' ] = str( 'Reactor' + ':' + \
#                                               aSystemPath + ':' + \
#                                                aChildElement.getAttribute( 'id' ) )
#                    anEntity[ 'Name' ]   = str( aChildElement.getAttribute( 'name' ) )
#
#                    anEntityEntityList.append( anEntity )
#        return anEntityEntityList





    def getEntityPropertyElementsList( self ):
        aPropertyList = self.getElementsByTagName( 'Property' )
        return aPropertyList


    def getElementsByTagName( self, aTagName ):
        anElementsList = self.__theDocument.getElementsByTagName( aTagName )
        return anElementsList




    ##---------------------------------------------
    ## Methods for Methods
    ##---------------------------------------------

    def createElement( self, aTagName ):
        """make an element"""
        aNewElement = self.__theDocument.createElement( aTagName )
        return aNewElement



    def createPropertyElement( self, aName, aValueList ):
        aPropertyElement = self.createElement( 'property' )
        aPropertyElement.setAttribute( 'name', aName )
        aPropertyElement = self.appendValueElements( aPropertyElement, aValueList )

        return aPropertyElement



    def appendValueElements( self, aMotherElement, aValueList ):
        aValueNumber = 0
        for aValue in aValueList:
            aValueElement = self.createElement( 'value' )
            aValueElement.setAttribute( 'number', str( aValueNumber ) )
            aValueData = self.__theDocument.createTextNode( aValue )
            aValueElement.appendChild( aValueData )
            aMotherElement.appendChild( aValueElement )

            aValueNumber = aValueNumber + 1

        return aMotherElement




    def asEntityInfo( self, aFullId ):
        aTargetEntity = {}
        aTargetEntity[ 'Type' ] = aFullId.split( ':' )[0]
        aTargetEntity[ 'Path' ] = aFullId.split( ':' )[1]
        aTargetEntity[ 'Id' ]   = aFullId.split( ':' )[2]
        
        return aTargetEntity



    def convertSystemFullId2SystemId( self, aSystemFullId ):
        """
        aSystemFullId : ex) System:/CELL:CYTOPLASM
        return -> aSystemId [string] : ex) /CELL/CYTOPLASM
        """

        aPathToSystem   = aSystemFullId.split( ':' )[1]
        aSystemSimpleId = aSystemFullId.split( ':' )[2]

        if( aSystemSimpleId == '/' ):
            aSystemId = '/'

        elif( aPathToSystem == '/' ):
            aSystemId = '/' +aSystemSimpleId

        else:
            aSystemId = aPathToSystem + '/' +aSystemSimpleId
            
        return aSystemId



    def convertSystemId2SystemFullId( self, aSystemId ):
        """
        aSystemId : ex) /CELL/CYTOPLASM
        return -> aSystemFullId [string] : ex) System:/CELL:CYTOPLASM
        """

        aSystemIdArray  = aSystemId.split( '/' )
        aSystemSimpleId = aSystemIdArray[-1]
        
        if ( aSystemId == '/' ):
            aSystemFullId = 'System::/'

        elif( len( aSystemIdArray ) == 2 ):
            aSystemFullId = 'System:/:' + aSystemSimpleId

        else:
            del aSystemIdArray[-1]
            aPathToSystem = string.join( aSystemIdArray, '/' )
            aSystemFullId = 'System:' + aPathToSystem + ':' + aSystemSimpleId

        return aSystemFullId



    def asSystemPath( self, aTargetSystem ):
        """convert fullid of system to fullpath
           ex.) System:/CELL:CYTOPLASM -> /CELL/CYTOPLASM
        """

        aSystemId = aTargetSystem.getAttribute( 'id' )
        aSystemPath = aSystemId
        return aSystemPath


#        if( aTargetSystem.getAttribute( 'id' ).split( ':' )[1] == '' ):
#            aSystemPath = '/'
#            
#        elif( aTargetSystem.getAttribute( 'id' ).split( ':' )[1] == '/' ):
#            aSystemPath = aTargetSystem.getAttribute( 'id' ).split( ':' )[1] + \
#                          aTargetSystem.getAttribute( 'id' ).split( ':' )[2]
#
#        else:
#            aSystemPath = aTargetSystem.getAttribute( 'id' ).split( ':' )[1] + \
#                          '/' + aTargetSystem.getAttribute( 'id' ).split( ':' )[2]
#
#        return aSystemPath



    def asPathToSystem( self, aFullPathOfSystem ):


        if( aFullPathOfSystem == '/' ):
            aPathToTargetSystem = ':/'

        else:
            aFullPathInfo = aFullPathOfSystem.split( '/' )
            aTargetSystemId = aFullPathInfo[-1]

            del aFullPathInfo[-1]
            aPathToTargetSystemInfo = aFullPathInfo

            if( len( aPathToTargetSystemInfo ) == 1 ):
                aPathToTargetSystem = '/:' + aTargetSystemId
            else:
                aPathToTargetSystem = str( string.join( aPathToTargetSystemInfo, '/' ) + ':' + aTargetSystemId )

        return aPathToTargetSystem
        
