"""
This is emllib for EML
"""
__author__     = 'suzuki'
__email__      = 'suzuki@sfc.keio.ac.jp'
__startDate__  = '020316'
__lastUpdate__ = '020911'

__Memo__ = '\
'

__Todo__ = '''
'''



#------------------- window line ---------------------------------------------#

from xml.dom import minidom

import string
import types

from types import *

from ecssupport import *

#---------------------------------------------------------"""
class Eml:

    
    def __init__( self, aFileObject=None ):
        """read EML file and make domtree"""

        if aFileObject is None:
            aStringData = '<?xml version="1.0" ?><eml></eml>'
        else:
            aStringData = string.join( string.join( aFileObject.readlines(), '' ).split( '\n' ), '' )

        self.__theDocument = minidom.parseString( aStringData )

        self.__clearCache()


    def asString( self ):
        """return domtree as string"""

        return self.__theDocument.toprettyxml(indent="", newl="\n")



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
        aStepperElement = self.__createElement( 'stepper' )
        aStepperElement.setAttribute( 'class', aClass )
        aStepperElement.setAttribute( 'id', anID )
        
        self.__theDocument.documentElement.childNodes.append( aStepperElement )


    

    def deleteStepper( self, anID ):
        """delete a stepper"""

        for anElement in self.__theDocument.firstChild.childNodes:
            if anElement.nodeName == 'stepper' and \
                   anElement.getAttribute( 'id' )    == anID:

                anElement.removeChild( aChildElement )
        

    


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

            if aChildNode.nodeName == 'property':

                aPropertyNode = aChildNode
                aPropertyName = aPropertyNode.getAttribute( 'name' )
                aPropertyList.append( str( aPropertyName ) )

        return aPropertyList




    def getStepperProperty( self, aStepperID, aPropertyName ):

        aValueList = []

        aStepperNode = self.__getStepperNode( aStepperID )
        for aChildNode in aStepperNode.childNodes:

            if aChildNode.nodeName == 'property':

                if aChildNode.getAttribute( 'name' ) == aPropertyName:

                    aPropertyNode = aChildNode

                    for aChildNode in aPropertyNode.childNodes:
                        if aChildNode.nodeName == 'value':

                            aValue = str( aChildNode.firstChild.nodeValue )
                            aValueList.append( aValue )
    
        return aValueList




    def getStepperClass( self, aStepperID ):

        aStepperNode = self.__getStepperNode( aStepperID )
        return aStepperNode.getAttribute( 'class' )



    def __getStepperNodeList( self ):
        """private"""

        aStepperNodeList = []

        for aTargetNode in self.__theDocument.documentElement.childNodes:
            if aTargetNode.nodeName == 'stepper':
                aStepperNode = aTargetNode
                aStepperNodeList.append( aStepperNode )

        return aStepperNodeList


    def setStepperProperty( self, aStepperID, aPropertyName, aValue ):

        aPropertyElement = self.__createPropertyNode( aPropertyName, aValue )
        aStepperNode = self.__getStepperNode( aStepperID )

        aStepperNode.appendChild( aPropertyElement )
        

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
        anEntityElement = self.__createElement( string.lower( anEntityType ) )
        anEntityElement.setAttribute( 'class', aClass )

        if( anEntityType == 'System' ):

            anID = self.convertSystemFullID2SystemID( aFullID )
            anEntityElement.setAttribute( 'id', anID )
            self.__theDocument.documentElement.appendChild( anEntityElement )

        elif( anEntityType == 'Variable' or anEntityType == 'Process' ):

            anID = aFullID.split( ':' )[2]
            anEntityElement.setAttribute( 'id', anID )

            aTargetFullPath = aFullID.split( ':' )[1]
            for aTargetNode in self.__theDocument.documentElement.childNodes:

                if aTargetNode.nodeName == 'system':
                    aTargetSystem = aTargetNode

                    aSystemFullPath = self.__asSystemPath( aTargetSystem )
               
                    if aTargetFullPath == aSystemFullPath:
                        aTargetSystem.appendChild( anEntityElement )

        self.__addToCache( aFullID, anEntityElement )



    def deleteEntity( self, aFullID ):
        """delete an entity"""

        aTargetEntity = asEntityInfo( aFullID )

        if aTargetEntity[ 'Type' ] == 'System':
            for anElement in self.__theDocument.firstChild.childNodes:

                if self.convertSystemID2SystemFullID( anElement.getAttribute( 'id' ) ) == aFullID:
                    self.__theDocument.firstChild.removeChild( anElement )

        else:
            for anElement in self.__theDocument.firstChild.childNodes:
                if anElement.nodeName == 'system':
                    if self.__asSystemPath( anElement ) == aTargetEntity[ 'Path' ]:

                        for aChild in anElement.childNodes:
                            if aChild.nodeName == string.lower( aTargetEntity[ 'Type' ] ) and \
                               aChild.getAttribute( 'id' ) == aTargetEntity[ 'ID' ]:

                                anElement.removeChild( aChild )

        self.__removeFromCache( aFullID )


    def isEntityExist( self, aFullID ):


        try:
            __getEntityNode( aFullID )
        except:
            return 0
        else:
            return 1


    def getEntityClass( self, aFullID ):
        anEntityNode = self.__getEntityNode( aFullID )

        return anEntityNode.getAttribute( 'class' )
        
        
    def setEntityProperty( self, aFullID, aPropertyName, aValueList ):

        anEntityPropertyElement = self.__createPropertyNode( aPropertyName,\
                                                             aValueList )
        
        aTargetNode = self.__getEntityNode( aFullID )
        aTargetNode.appendChild( anEntityPropertyElement )



    def deleteEntityProperty( self, aFullID, aPropertyName ):

        aTargetNode = self.__getEntityNode( aFullID )

        for aChild in aTargetNode.childNodes:
            if aChild.nodeName == 'property' and\
                   aChild.getAttribute( 'name' ) == aPropertyName:

                aTargetSystem.removeChild( aPropertyNode )


    def getEntityList( self, anEntityType, aSystemPath ):

        # better if this method creates entity cache on the fly?

        aType = string.lower( anEntityType )

        if aType == 'system':

            anEntityList = self.__getSystemList( aSystemPath )

        else:
            aSystemNodeList = self.__theDocument.getElementsByTagName( 'system' )

            anEntityList = []

            for aSystemNode in aSystemNodeList:

                if aSystemNode.getAttribute( 'id' ) == aSystemPath:
                    
                    for aChildNode in aSystemNode.childNodes:
                        
                        if aChildNode.nodeName == aType:
                            
                            anEntityList.append( str( aChildNode.getAttribute( 'id' ) ) )


        return anEntityList



    def getEntityPropertyList( self, aFullID ):

        anEntityNode = self.__getEntityNode( aFullID )
        anEntityPropertyList = []

        for aChildNode in anEntityNode.childNodes:

            if aChildNode.nodeName == 'property':

                anEntityPropertyList.append( str( aChildNode.getAttribute( 'name' ) ) )

        return anEntityPropertyList



    def getEntityProperty( self, aFullPNString ):


        aFullPN = createFullPN( aFullPNString )
        aPropertyName = aFullPN[3]
        aFullID = createFullIDString( convertFullPNToFullID( aFullPN ) )
        anEntityPropertyNode = self.__getEntityPropertyNode( aFullID, aPropertyName )

        return self.__createValueList( anEntityPropertyNode )
                




    ##-------------------------------------------
    ## Cache manipulations
    ##-------------------------------------------

    def __findInCache( self, aFullID ):

        return self.__entityNodeCache[ aFullID ]

    def __addToCache( self, aFullID, aNode ):

        self.__entityNodeCache[ aFullID ] = aNode

    def __removeFromCache( self, aFullID ):

        del self.__entityNodeCache[ aFullID ]

    def __clearCache( self ):

        self.__entityNodeCache = {}



    ##-------------------------------------------
    ## Utils
    ##-------------------------------------------

    def __createValueList( self, aValueNode ):

        if aValueNode.firstChild.nodeType == minidom.Node.TEXT_NODE:

            return str( aValueNode.firstChild.nodeValue )

        elif aValueNode.firstChild.nodeType == minidom.Node.ELEMENT_NODE:

            aValueList = []
            for aChildNode in aValueNode.childNodes:
                if aChildNode.nodeName == 'value':
                    aValueList.append( self.__createValueList( aChildNode ) )

            return aValueList


    def __getSystemList( self, aSystemPath ):

        aSystemList = []
        aSystemNodeList = self.__theDocument.getElementsByTagName( 'system' )

        if aSystemPath == '':
            for aSystemNode in aSystemNodeList:

                aSystemID = str( aSystemNode.getAttribute( 'id' ) )
                if( aSystemID == '/' ):
                    return [ aSystemID, ]


        aSystemPathLength = len( aSystemPath )

        for aSystemNode in aSystemNodeList:

            aSystemID = str( aSystemNode.getAttribute( 'id' ) )

            if len( aSystemID ) > len( aSystemPath ) and\
                   string.find( aSystemID, aSystemPath ) == 0 and\
                   aSystemID[aSystemPathLength+1:].find( '/' ) == -1:
                aSystemList.append( string.split( aSystemID, '/' )[-1] )

        return aSystemList


    def __getEntityNode( self, aFullID ):
        
        # first look up the cache
        try:
            return self.__findInCache( aFullID )
        except:
            pass

        aSystemNodeList = self.__theDocument.getElementsByTagName( 'system' )
        anEntityInfo = self.asEntityInfo( aFullID )
        
        aSystemPath = anEntityInfo[ 'Path' ]
        aType = anEntityInfo[ 'Type' ]
        anID = anEntityInfo[ 'ID' ]

        if anEntityInfo[ 'Type' ] == 'System':
            aSystemPath = joinSystemPath( aSystemPath, anID )

            for aSystemNode in aSystemNodeList:
                
                if aSystemNode.getAttribute( 'id' ) == aSystemPath:
                    self.__addToCache( aFullID, aSystemNode )
                    return aSystemNode

        else:
            for aSystemNode in aSystemNodeList:
                
                if aSystemNode.getAttribute( 'id' ) == aSystemPath:

                    for aChildNode in aSystemNode.childNodes:
                        
                        if string.capwords( aChildNode.nodeName ) == aType and\
                               aChildNode.getAttribute( 'id' ) == anID:

                            self.__addToCache( aFullID, aChildNode )
                            return aChildNode


        raise "Entity [" + aFullID + "] not found."

                        



    def __getEntityPropertyNode( self, aFullID, aPropertyName ):

        anEntityNode = self.__getEntityNode( aFullID )

        anEntityInfo = self.asEntityInfo( aFullID )


        # what if multiple propety elements with the same name exist?
        for aChildNode in anEntityNode.childNodes:

            if aChildNode.nodeName == 'property':

                aPropertyNode = aChildNode
                if aPropertyNode.getAttribute( 'name' ) == aPropertyName:

                    return aPropertyNode






    ##---------------------------------------------
    ## Methods for Methods
    ##---------------------------------------------

    def __createElement( self, aTagName ):
        """make an element"""
        return self.__theDocument.createElement( aTagName )


    def __createPropertyNode( self, aPropertyName, aValueList ):

        aPropertyElement = self.__createElement( 'property' )
        aPropertyElement.setAttribute( 'name', aPropertyName )

        for aValue in aValueList:
            aValueNode = self.__createValueNode( aValue )

            if aValueNode:
                aPropertyElement.appendChild( aValueNode )

        return aPropertyElement
    

    def __createValueNode( self, aValue ):

        if type( aValue ) is types.TupleType or \
               type( aValue ) is types.ListType:    # vector value

            aValueNode = self.__createElement( 'value' )

            for anItem in aValue:
                aChildValueData = self.__createValueNode( anItem )
                aValueNode.appendChild( aChildValueData )

            return aValueNode

        else:        # scaler value
 
            aValueNode = self.__createElement( 'value' )
            aValueData = self.__theDocument.createTextNode( aValue )
            aValueNode.appendChild( aValueData )

            return aValueNode



    def asEntityInfo( self, aFullID ):
        aTargetEntity = {}
        aParsedFullID = aFullID.split( ':' )
        aTargetEntity[ 'Type' ] = aParsedFullID[0]
        aTargetEntity[ 'Path' ] = aParsedFullID[1]
        aTargetEntity[ 'ID' ]   = aParsedFullID[2]

        return aTargetEntity



    def convertSystemFullID2SystemID( self, aSystemFullID ):
        """
        aSystemFullID : ex) System:/CELL:CYTOPLASM
        return -> aSystemID [string] : ex) /CELL/CYTOPLASM
        """
        aParsedSystemFullID = aSystemFullID.split( ':' )

        aPathToSystem   = aParsedSystemFullID[1]
        aSystemSimpleID = aParsedSystemFullID[2]

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

        aParsedSystemID  = aSystemID.split( '/' )
        aSystemSimpleID = aParsedSystemID[-1]
        
        if ( aSystemID == '/' ):
            aSystemFullID = 'System::/'

        elif( len( aParsedSystemID ) == 2 ):
            aSystemFullID = 'System:/:' + aSystemSimpleID

        else:
            del aParsedSystemID[-1]
            aPathToSystem = string.join( aParsedSystemID, '/' )
            aSystemFullID = 'System:' + aPathToSystem + ':' + aSystemSimpleID

        return aSystemFullID


    def __asSystemPath( self, aTargetSystem ):
        """convert fullid of system to fullpath
           ex.) System:/CELL:CYTOPLASM -> /CELL/CYTOPLASM
        """

        aSystemID = aTargetSystem.getAttribute( 'id' )
        aSystemPath = aSystemID
        return aSystemPath

