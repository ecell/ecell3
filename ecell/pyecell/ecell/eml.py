"""
This is emllib for EML
"""
__author__     = 'suzuki'
__email__      = 'suzuki@sfc.keio.ac.jp'
__startDate__  = '020316'
__lastUpdate__ = '020911'

__Memo__ = '\
'

__Todo__ = '\
- How to use multi stepper? Only one stepper can be usable. [020708]\
+ How to overwrite Property? Think about values. [020701] finished'


#------------------- window line ------------------------------------------------------------------#

from xml.dom import minidom

import string

from types import *

from ecssupport import *

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

    def createStepper( self, aClass, anID ):
        """create a stepper"""
        aStepperElement = self.createElement( 'stepper' )
        aStepperElement.setAttribute( 'class', aClass )
        aStepperElement.setAttribute( 'id', anID )

        
        for aTargetNode in self.__theDocument.documentElement.childNodes:
            if aTargetNode.tagName == 'stepperlist':
                aTargetNode.childNodes.append( aStepperElement )


    

    def deleteStepper( self, anID ):
        """delete a stepper"""

        for anElement in self.__theDocument.firstChild.childNodes:
            if anElement.tagName == 'stepperlist':

                aStepperlistNode = anElement
                for aChildElement in aStepperlistNode.childNodes:
                    if aChildElement.tagName == 'stepper' and \
                       aChildElement.getAttribute( 'id' )    == anID:

                        aStepperlistNode.removeChild( aChildElement )
        

    


    def getStepperList( self ):

        aStepperNodeList = self.__getStepperNodeList()

        aStepperList = []
        for aTargetStepperNode in aStepperNodeList:

            aStepperID = aTargetStepperNode.getAttribute( 'id' )
            aStepperList.append( str( aStepperID ) )

        return aStepperList




    def getStepperPropertyList( self, aStepperID ):

        aStepperNodeList = self.__getStepperNode( aStepperID )
        aPropertyList = []

        for aChildNode in aStepperNodeList.childNodes:

            if aChildNode.tagName == 'property':

                aPropertyNode = aChildNode
                aPropertyName = aPropertyNode.getAttribute( 'name' )
                aPropertyList.append( str( aPropertyName ) )

        return aPropertyList




    def getStepperProperty( self, aStepperID, aPropertyName ):

        aValueList = []

        aStepperNode = self.__getStepperNode( aStepperID )
        for aChildNode in aStepperNode.childNodes:

            if aChildNode.tagName == 'property':

                if aChildNode.getAttribute( 'name' ) == aPropertyName:

                    aPropertyNode = aChildNode

                    for aChildNode in aPropertyNode.childNodes:
                        if aChildNode.tagName == 'value':

                            aValue = aChildNode.firstChild.toxml()
                            aValueList.append( aValue )
    
        return aValueList




    def getStepperClass( self, aStepperID ):

        aStepperNode = self.__getStepperNode( aStepperID )
        return aStepperNode.getAttribute( 'class' )



    def __getStepperNodeList( self ):
        """private"""

        aStepperNodeList = []

        for aTargetNode in self.__theDocument.documentElement.childNodes:
            if aTargetNode.tagName == 'stepperlist':
                aStepperlistElement = aTargetNode

                for aTargetNode in aStepperlistElement.childNodes:
                    if aTargetNode.tagName == 'stepper':
                        aStepperNode = aTargetNode
                        aStepperNodeList.append( aStepperNode )

        return aStepperNodeList




    def __getStepperNode( self, aStepperID ):
        """private"""
        
        aStepperNodeList = self.__getStepperNodeList()

        for aTargetStepperNode in aStepperNodeList:
            if aTargetStepperNode.getAttribute( 'id' ) == aStepperID:
                
                return aTargetStepperNode
            





    ##---------------------------------------------
    ## Methods for Entity
    ##---------------------------------------------

    def createEntity( self, aClass, aFullID ):

        anEntityType = self.asEntityInfo( aFullID )[ 'Type' ]
        anEntityElement = self.createElement( string.lower( anEntityType ) )
        anEntityElement.setAttribute( 'class', aClass )

        if( anEntityType == 'System' ):

            anID = self.convertSystemFullID2SystemID( aFullID )
            anEntityElement.setAttribute( 'id', anID )
            self.__theDocument.documentElement.appendChild( anEntityElement )

        elif( anEntityType == 'Substance' or anEntityType == 'Reactor' ):

            anID = aFullID.split( ':' )[2]
            anEntityElement.setAttribute( 'id', anID )

            aTargetFullPath = aFullID.split( ':' )[1]
            for aTargetNode in self.__theDocument.documentElement.childNodes:

                if aTargetNode.tagName == 'system':
                    aTargetSystem = aTargetNode

                    aSystemFullPath = self.asSystemPath( aTargetSystem )
               
                    if aTargetFullPath == aSystemFullPath:
                        aTargetSystem.appendChild( anEntityElement )




    def deleteEntity( self, aFullID ):
        """delete an entity"""

        aTargetEntity = self.asEntityInfo( aFullID )

        if aTargetEntity[ 'Type' ] == 'System':
            for anElement in self.__theDocument.firstChild.childNodes:

                if self.convertSystemID2SystemFullID( anElement.getAttribute( 'id' ) ) == aFullID:
                    self.__theDocument.firstChild.removeChild( anElement )

        else:
            for anElement in self.__theDocument.firstChild.childNodes:
                if anElement.tagName == 'system':
                    if self.asSystemPath( anElement ) == aTargetEntity[ 'Path' ]:

                        for aChild in anElement.childNodes:
                            if aChild.tagName == string.lower( aTargetEntity[ 'Type' ] ) and \
                               aChild.getAttribute( 'id' ) == aTargetEntity[ 'ID' ]:

                                anElement.removeChild( aChild )
                        


    def isEntityExist( self, aFullID ):

        aTargetEntity = self.asEntityInfo( aFullID )

        anExistence = 0

        aSystemList = self.__theDocument.getElementsByTagName( 'system' )
        for aSystem in aSystemList:

            ## for System
            if aTargetEntity[ 'Type' ] == 'System':
                aSystemPath = aSystem.getAttribute( 'id' ).split( ':' )[1]
                if aSystemPath == aTargetEntity[ 'Path' ]:
                    anExistence = 1

            ## for Substance or Reactor
            else:
                aSystemPath = self.asSystemPath( aSystem )

                if aSystemPath == aTargetEntity[ 'Path' ]:
                    for anElement in aSystem.childNodes:

                        if anElement.tagName == string.lower( aTargetEntity[ 'Type' ] ) and \
                           anElement.getAttribute( 'id' ) == aTargetEntity[ 'ID' ]:

                            anExistence = 1
                            
        return anExistence



    def getEntityClass( self, aFullID ):
        anEntityNode = self.__getEntityNode( aFullID )

        return anEntityNode.getAttribute( 'class' )
        
        


    ##---------------------------------------------
    ## Methods for Property
    ##---------------------------------------------

    def setEntityProperty( self, aFullID, aPropertyName, aValueList ):

        anEntityID   = self.asEntityInfo( aFullID )[ 'ID' ]
        anEntityType = self.asEntityInfo( aFullID )[ 'Type' ]
        anEntityPath = self.asEntityInfo( aFullID )[ 'Path' ]

        anEntityPropertyElement = self.createPropertyNode( aPropertyName, aValueList )
        
        for aTargetNode in self.__theDocument.firstChild.childNodes:
            if aTargetNode.tagName == 'system':

                aTargetSystem       = aTargetNode
                aTargetSystemID     = aTargetSystem.getAttribute( 'id' )
                aTargetSystemFullID = self.convertSystemID2SystemFullID( aTargetSystemID )


                if anEntityType == 'System':

                    aTargetSystem = aTargetNode
                    aTargetSystemID     = aTargetSystem.getAttribute( 'id' )
                    aTargetSystemFullID = self.convertSystemID2SystemFullID( aTargetSystemID )

                    if aTargetSystemFullID == aFullID:
                        aTargetSystem.appendChild( anEntityPropertyElement )


                elif aTargetSystemID == anEntityPath:
                    for aTargetChildNode in aTargetSystem.childNodes:

                        if string.capwords( aTargetChildNode.tagName ) == anEntityType:

                            if aTargetChildNode.getAttribute( 'id' ) == anEntityID:

                                aTargetChildNode.appendChild( anEntityPropertyElement )
                    

                    


    def deleteEntityProperty( self, aFullID, aPropertyName ):

        ## aPropertyElement = self.createPropertyNode( aName, aValueList )

        anEntityID   = self.asEntityInfo( aFullID )[ 'ID' ]
        anEntityType = self.asEntityInfo( aFullID )[ 'Type' ]
        anEntityPath = self.asEntityInfo( aFullID )[ 'Path' ]

        for aTargetNode in self.__theDocument.firstChild.childNodes:
            if aTargetNode.tagName == 'system':

                aTargetSystem       = aTargetNode
                aTargetSystemID     = aTargetSystem.getAttribute( 'id' )
                aTargetSystemFullID = self.convertSystemID2SystemFullID( aTargetSystemID )

                if anEntityType == 'System':

                    aTargetSystem = aTargetNode
                    aTargetSystemID     = aTargetSystem.getAttribute( 'id' )
                    aTargetSystemFullID = self.convertSystemID2SystemFullID( aTargetSystemID )

                    if aTargetSystemFullID == aFullID:

                        for aTargetChild in aTargetSystem.childNodes:
                            if aTargetChild.tagName == 'property':

                                aPropertyNode = aTargetChild
                                if aPropertyNode.getAttribute( 'name' ) == aPropertyName:

                                    aPropertyNodeToDelete = aPropertyNode

                        aTargetSystem.removeChild( aPropertyNode )



                elif aTargetSystemID == anEntityPath:

                    for aTargetChildNode in aTargetSystem.childNodes:

                        if string.capwords( aTargetChildNode.tagName ) == anEntityType:

                            if aTargetChildNode.getAttribute( 'id' ) == anEntityID:

                                aTargetEntityNode = aTargetChildNode
                                for aTargetChild in aTargetEntityNode.childNodes:
                                    
                                    if aTargetChild.tagName == 'property':
                                        
                                        aPropertyNode = aTargetChild

                                        if aPropertyNode.getAttribute( 'name' ) == aPropertyName:

                                            aPropertyNodeToDelete = aPropertyNode
                                aTargetEntityNode.removeChild( aPropertyNodeToDelete )
                               
                    




    ##---------------------------------------------
    ## Methods for Read
    ##---------------------------------------------


    def getEntityList( self, anEntityType, aSystemPath ):

        if anEntityType == 'System':

            anEntityList = self.__getSystemList( anEntityType, aSystemPath )


        else:
            aSystemNodeList = self.__theDocument.getElementsByTagName( 'system' )

            anEntityList = []

            for aSystemNode in aSystemNodeList:

                if aSystemNode.getAttribute( 'id' ) == aSystemPath:
                    
                    for aChildNode in aSystemNode.childNodes:
                        
                        if string.capwords( aChildNode.tagName ) == anEntityType:
                            
                            anEntityList.append( str( aChildNode.getAttribute( 'id' ) ) )

        return anEntityList






    def getEntityPropertyList( self, aFullID ):

        anEntityNode = self.__getEntityNode( aFullID )
        anEntityPropertyList = []

        for aChildNode in anEntityNode.childNodes:

            if aChildNode.tagName == 'property':

                anEntityPropertyList.append( str( aChildNode.getAttribute( 'name' ) ) )

        return anEntityPropertyList





    def getEntityProperty( self, aFullPNString ):


        aFullPN = createFullPN( aFullPNString )
        aPropertyName = aFullPN[3]
        aFullID = createFullIDString( convertFullPNToFullID( aFullPN ) )
        anEntityPropertyNode = self.__getEntityPropertyNode( aFullID, aPropertyName )

        aPropertyValueList = []
        for aChildNode in anEntityPropertyNode.childNodes:

            if aChildNode.tagName == 'value':

                aPropertyValueList.append( self.__createValueList( aChildNode ) )
        return aPropertyValueList
                


    def __createValueList( self, aValueNode ):
        if aValueNode.firstChild.nodeType == minidom.Node.TEXT_NODE:

            return aValueNode.firstChild.toxml()

        elif aValueNode.firstChild.nodeType == minidom.Node.ELEMENT_NODE:
            
            aValueList = []

            for aChildNode in aValueNode.childNodes:
                aValueList.append( self.__createValueList( aChildNode ) )

            return aValueList

    def __getSystemList( self, anEntityType, aSystemPath ):


        aSystemList = []
        aSystemNodeList = self.__theDocument.getElementsByTagName( 'system' )

        if aSystemPath == '':
            aSystemPath = '/'

        aSystemPathLength = len( aSystemPath )

        for aSystemNode in aSystemNodeList:

            aSystemID = str( aSystemNode.getAttribute( 'id' ) )

            if len( aSystemID ) > len( aSystemPath ) and\
                   string.find( aSystemID, aSystemPath ) == 0:
                aRelativePath = aSystemID[aSystemPathLength:]
                if aRelativePath[0] == '/':
                    aRelativePath = aRelativePath[1:]
                if string.find( aRelativePath, '/' ) == -1:
                    aSystemList.append( aSystemID )

        return aSystemList


    def __getEntityNode( self, aFullID ):
        
        aSystemNodeList = self.__theDocument.getElementsByTagName( 'system' )
        anEntityInfo = self.asEntityInfo( aFullID )
        
        
        if anEntityInfo[ 'Type' ] == 'System':
            aSystemPath = joinSystemPath( anEntityInfo[ 'Path' ],\
                                          anEntityInfo[ 'ID' ] )

            for aSystemNode in aSystemNodeList:
                
                if aSystemNode.getAttribute( 'id' ) == aSystemPath:
                    return aSystemNode

        else:
            aSystemPath = anEntityInfo[ 'Path' ]

            for aSystemNode in aSystemNodeList:
                
                if aSystemNode.getAttribute( 'id' ) == aSystemPath:

                    for aChildNode in aSystemNode.childNodes:
                        
                        if string.capwords( aChildNode.tagName ) == anEntityInfo[ 'Type' ]:

                            if aChildNode.getAttribute( 'id' ) == anEntityInfo[ 'ID' ]:

                                return aChildNode

        raise "Entity [" + aFullID + "] not found."

                        



    def __getEntityPropertyNode( self, aFullID, aPropertyName ):

        anEntityNode = self.__getEntityNode( aFullID )

        anEntityInfo = self.asEntityInfo( aFullID )

        for aChildNode in anEntityNode.childNodes:

            if aChildNode.tagName == 'property':

                aPropertyNode = aChildNode
                if aPropertyNode.getAttribute( 'name' ) == aPropertyName:

                    return aPropertyNode






    def getPropertyList( self ):
        """OLD METHOD"""

        aSystemPropertyList  = self.getSystemPropertyList()
        aSubstanceOrReactorPropertyList = self.getSubstanceOrReactorPropertyList()

        aPropertyList = aSystemPropertyList + aSubstanceOrReactorPropertyList

        return aPropertyList        









        
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
            aSystem[ 'FullID' ] = str( 'System:' \
                                       + self.asPathToSystem( aSystemElement.getAttribute( 'id' ) ) )
            aSystem[ 'Name' ]   = str( aSystemElement.getAttribute( 'name' ) )
            
            if not aSystem[ 'FullID' ] == 'System::/':
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
                    anEntity[ 'FullID' ] = str( string.capwords( aChildElement.tagName ) + ':' + \
                                                aSystemPath + ':' + \
                                                aChildElement.getAttribute( 'id' ) )
                    anEntity[ 'Name' ]   = str( aChildElement.getAttribute( 'name' ) )
                    
                    anEntityEntityList.append( anEntity )    
        return anEntityEntityList







    def getPropertyElementsList( self ):
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




    def createPropertyNode( self, aPropertyName, aValueList ):

        aPropertyElement = self.createElement( 'property' )
        aPropertyElement.setAttribute( 'name', aPropertyName )


        for aValue in aValueList:
            aValueNode = self.createValueNode( aValue )

            if aValueNode:
                aPropertyElement.appendChild( aValueNode )

        return aPropertyElement
    



    def createValueNode( self, aValue ):

        if isinstance( aValue, StringType ):
            aValueNode = self.createElement( 'value' )
            aValueData = self.__theDocument.createTextNode( aValue )
            aValueNode.appendChild( aValueData )

            return aValueNode


        elif aValue: ## NOW 2-dimension only
            
            aValueList = aValue

            aValueNode = self.createElement( 'value' )

            for aValue in aValueList:
                aChildValueNode = self.createElement( 'value' )
                aChildValueData = self.__theDocument.createTextNode( aValue )
                aChildValueNode.appendChild( aChildValueData )
                
                aValueNode.appendChild( aChildValueNode )

            return aValueNode





    def appendValueElements( self, aMotherElement, aValueList ):

        for aValue in aValueList:

            anIsString = isinstance( aValueList, StringType )

            if anIsString:
                        
                aValueElement = self.createElement( 'value' )
                aValueData = self.__theDocument.createTextNode( aValue )
                aValueElement.appendChild( aValueData )
                aMotherElement.appendChild( aValueElement )

            else:
                pass
                
        return aMotherElement





    def asEntityInfo( self, aFullID ):
        aTargetEntity = {}
        aTargetEntity[ 'Type' ] = aFullID.split( ':' )[0]
        aTargetEntity[ 'Path' ] = aFullID.split( ':' )[1]
        aTargetEntity[ 'ID' ]   = aFullID.split( ':' )[2]
        
        return aTargetEntity





    def convertSystemFullID2SystemID( self, aSystemFullID ):
        """
        aSystemFullID : ex) System:/CELL:CYTOPLASM
        return -> aSystemID [string] : ex) /CELL/CYTOPLASM
        """

        aPathToSystem   = aSystemFullID.split( ':' )[1]
        aSystemSimpleID = aSystemFullID.split( ':' )[2]

        if( aSystemSimpleID == '/' ):
            aSystemID = '/'

        elif( aPathToSystem == '/' ):
            aSystemID = '/' +aSystemSimpleID

        else:
            aSystemID = aPathToSystem + '/' +aSystemSimpleID
            
        return aSystemID





    def convertSystemID2SystemFullID( self, aSystemID ):
        """
        aSystemID : ex) /CELL/CYTOPLASM
        return -> aSystemFullID [string] : ex) System:/CELL:CYTOPLASM
        """

        aSystemIDArray  = aSystemID.split( '/' )
        aSystemSimpleID = aSystemIDArray[-1]
        
        if ( aSystemID == '/' ):
            aSystemFullID = 'System::/'

        elif( len( aSystemIDArray ) == 2 ):
            aSystemFullID = 'System:/:' + aSystemSimpleID

        else:
            del aSystemIDArray[-1]
            aPathToSystem = string.join( aSystemIDArray, '/' )
            aSystemFullID = 'System:' + aPathToSystem + ':' + aSystemSimpleID

        return aSystemFullID





    def asSystemPath( self, aTargetSystem ):
        """convert fullid of system to fullpath
           ex.) System:/CELL:CYTOPLASM -> /CELL/CYTOPLASM
        """

        aSystemID = aTargetSystem.getAttribute( 'id' )
        aSystemPath = aSystemID
        return aSystemPath





    def asPathToSystem( self, aFullPathOfSystem ):


        if( aFullPathOfSystem == '/' ):
            aPathToTargetSystem = ':/'

        else:
            aFullPathInfo = aFullPathOfSystem.split( '/' )
            aTargetSystemID = aFullPathInfo[-1]

            del aFullPathInfo[-1]
            aPathToTargetSystemInfo = aFullPathInfo

            if( len( aPathToTargetSystemInfo ) == 1 ):
                aPathToTargetSystem = '/:' + aTargetSystemID
            else:
                aPathToTargetSystem = str( string.join( aPathToTargetSystemInfo, '/' ) + ':' + aTargetSystemID )

        return aPathToTargetSystem
        
