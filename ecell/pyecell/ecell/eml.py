#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2007 Keio University
#       Copyright (C) 2005-2007 The Molecular Sciences Institute
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-Cell System is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
# 
# E-Cell System is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public
# License along with E-Cell System -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
# 
#END_HEADER
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
            aStringData = string.join(  map( string.strip, aFileObject.readlines()), '' )


        # minidom.parseString() is much faster than minidom.parse().. why?
        self.__theDocument = minidom.parseString( aStringData )

        for aNode in self.__theDocument.childNodes:

            if str( aNode.nodeName ) == '#comment':
                self.__theComment = aNode.nodeValue
            elif str( aNode.nodeName ) == 'eml':
                self.__theEmlNode = aNode
            else:
                pass
        

#        self.__clearCache()
        self.__reconstructCache()


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

        for anElement in self.__theEmlNode.childNodes:
            if anElement.nodeName == 'stepper' and \
                   anElement.getAttribute( 'id' ) == anID:

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
        for aPropertyNode in aStepperNode.childNodes:

            if aPropertyNode.nodeName == 'property':

                if aPropertyNode.getAttribute( 'name' ) == aPropertyName:

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

        # what if a property with the same name already exist?
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

        ( anEntityType, aTargetPath, anID ) = aFullID.split( ':' )
        anEntityElement = self.__createElement( anEntityType.lower() )
        anEntityElement.setAttribute( 'class', aClass )


        if( anEntityType == 'System' ):

            if aTargetPath != '':  # check if the supersystem exists
                dummy = self.__getSystemNode( aTargetPath )

            anID = convertSystemFullID2SystemID( aFullID )
            anEntityElement.setAttribute( 'id', anID )
            self.__theDocument.documentElement.appendChild( anEntityElement )

        elif( anEntityType == 'Variable' or anEntityType == 'Process' ):

            anEntityElement.setAttribute( 'id', anID )
            aTargetSystemNode = self.__getSystemNode( aTargetPath )
            aTargetSystemNode.appendChild( anEntityElement )

        else:            
            raise "unexpected error. %s should be System, Variable, or Process."%anEntityType

        self.__addToCache( aFullID, anEntityElement )



    def deleteEntity( self, aFullID ):
        """delete an entity"""

        aType = aFullID.split( ':', 1 )[0]

        if aType == 'System':
            for anElement in self.__theEmlNode.childNodes:

                if convertSystemID2SystemFullID( anElement.getAttribute( 'id' ) ) == aFullID:
                    self.__theEmlNode.removeChild( anElement )

        else:
            for anElement in self.__theEmlNode.childNodes:
                if anElement.nodeName == 'system':
                    if anElement.getAttribute( 'id' ) == aTargetEntity[ 'Path' ]:

                        for aChild in anElement.childNodes:
                            if aChild.nodeName == aTargetEntity[ 'Type' ].lower() and \
                               aChild.getAttribute( 'id' ) == aTargetEntity[ 'ID' ]:

                                anElement.removeChild( aChild )

        self.__removeFromCache( aFullID )


    def isEntityExist( self, aFullID ):

        try:
            self.__getEntityNode( aFullID )
        except:
            return 0
        else:
            return 1


    def getEntityClass( self, aFullID ):
        
        anEntityNode = self.__getEntityNode( aFullID )
 
        return str( anEntityNode.getAttribute( 'class' ) )
        
        
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

                aTargetNode.removeChild( aChild )


    def getEntityList( self, anEntityType, aSystemPath ):

        # better if this method creates entity cache on the fly?

        aType = anEntityType.lower()

        if aType == 'system':

            anEntityList = self.__getSubSystemList( aSystemPath )

        else:

            anEntityList = []

            for aSystemNode in self.__theEmlNode.childNodes:
                if aSystemNode.nodeName == 'system' and \
                       aSystemNode.getAttribute( 'id' ) == aSystemPath:
                    
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

    def setEntityInfo( self, aFullID, InfoStrings ):

        anEntityInfoElement = self.__createInfoNode( InfoStrings )
        aTargetNode = self.__getEntityNode( aFullID )
        aTargetNode.appendChild( anEntityInfoElement )

    def getEntityInfo( self, aFullID ):

        anEntityInfoNode = self.__getEntityInfoNode( aFullID )

        return self.__createValueList( anEntityInfoNode )
    
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

    def __reconstructCache( self ):

        self.__clearCache()

        for aSystemNode in self.__theEmlNode.childNodes:
            if aSystemNode.nodeName == 'system':

                aSystemPath = aSystemNode.getAttribute( 'id' )
                aSystemFullID = convertSystemID2SystemFullID( aSystemPath )
                self.__addToCache( aSystemFullID, aSystemNode )

                for aChildNode in aSystemNode.childNodes:
                    aType = string.capwords( aChildNode.nodeName )

                    if  aType in ( 'Variable', 'Process' ):

                        anID = aChildNode.getAttribute( 'id' )
                        aFullID = aType + ':' + aSystemPath + ':' + anID
                        self.__addToCache( aFullID, aChildNode )

    ##-------------------------------------------
    ## Utils
    ##-------------------------------------------

    def __createValueList( self, aValueNode ):

        aNode = aValueNode.firstChild
        aNodeType = aNode.nodeType

        if aNodeType == aValueNode.TEXT_NODE:

            aValue = string.replace( str( aNode.nodeValue ), '#x0A', '\n')
            return aValue

        elif aNodeType == aValueNode.ELEMENT_NODE:

            return map( self.__createValueList, aValueNode.childNodes )

        else:
            raise "unexpected error."

    def __getSubSystemList( self, aSystemPath ):

        aTargetPath = aSystemPath.split( '/' )
        aTargetPathLength = len( aTargetPath )

        # if '' is given, return the root system ('/')
        if aTargetPathLength == 1:
            for aSystemNode in self.__theEmlNode.childNodes:
                if aSystemNode.nodeName == 'system' and \
                     aSystemNode.getAttribute( 'id' ) == '/':
                    return [ '/', ]
            return []


        aSystemList = []

        if aTargetPath[-1] == '':
            aTargetPath = aTargetPath[:-1]
            aTargetPathLength -= 1

        for aSystemNode in self.__theEmlNode.childNodes:
            if aSystemNode.nodeName == 'system':

                aSystemPath = str( aSystemNode.getAttribute( 'id' ) ).split( '/' )
                if aSystemPath[-1] == '':
                    aSystemPath = aSystemPath[:-1]
                    
                if len( aSystemPath ) == aTargetPathLength + 1 and \
                       aSystemPath[:aTargetPathLength] == aTargetPath:
                    aSystemList.append( aSystemPath[-1] )

        return aSystemList


    def __getEntityNode( self, aFullID ):
        
        # first look up the cache
        try:
            return self.__findInCache( aFullID )
        except:
            pass

        aType, aSystemPath, anID = aFullID.split( ':' )

        if aType == 'System':
            aSystemPath = joinSystemPath( aSystemPath, anID )
            return self.__getSystemNode( aSystemPath )
            
        aSystemNode = self.__getSystemNode( aSystemPath )

        for aChildNode in aSystemNode.childNodes:
                        
            if string.capwords( aChildNode.nodeName ) == aType and\
                   aChildNode.getAttribute( 'id' ) == anID:

                self.__addToCache( aFullID, aChildNode )
                return aChildNode

        raise "Entity [%s] not found."%aFullID

                        
    def __getSystemNode( self, aSystemPath ):

        aFullID = convertSystemID2SystemFullID( aSystemPath )

        # first look up the cache
        try:
            return self.__findInCache( aFullID )
        except:
            pass

        for aSystemNode in self.__theEmlNode.childNodes:
            print aSystemNode

            if aSystemNode.nodeName == 'system' and \
                   str( aSystemNode.getAttribute( 'id' ) ) == aSystemPath:
                self.__addToCache( aFullID, aSystemNode )
                return aSystemNode

        raise "System [%s] not found."%aFullID

    def __getEntityPropertyNode( self, aFullID, aPropertyName ):

        anEntityNode = self.__getEntityNode( aFullID )

        # what if multiple propety elements with the same name exist?
        for aChildNode in anEntityNode.childNodes:

            if aChildNode.nodeName == 'property' and \
                   aChildNode.getAttribute( 'name' ) == aPropertyName:

                return aChildNode



    def __getEntityInfoNode( self, aFullID ):

        anEntityNode = self.__getEntityNode( aFullID )

        for aChildNode in anEntityNode.childNodes:

            if aChildNode.nodeName == 'info':
                
                return aChildNode



    ##---------------------------------------------
    ## Methods for Methods
    ##---------------------------------------------

    def __createElement( self, aTagName ):
        """make an element"""
        return self.__theDocument.createElement( aTagName )


    def __createPropertyNode( self, aPropertyName, aValueList ):

        aPropertyElement = self.__createElement( 'property' )
        aPropertyElement.setAttribute( 'name', aPropertyName )

        map( aPropertyElement.appendChild,\
             map( self.__createValueNode, aValueList ) )

        return aPropertyElement


    def __createValueNode( self, aValue ):

        aValueNode = self.__createElement( 'value' )

        if type( aValue ) in ( types.TupleType, types.ListType ):    # vector value

            map( aValueNode.appendChild,\
                 map( self.__createValueNode, aValue ) )

        else:        # scaler value

            aNormalizedValue =  string.replace( aValue, '\n', '#x0A' )

            aValueData = self.__theDocument.createTextNode( aNormalizedValue )
            aValueNode.appendChild( aValueData )


        return aValueNode

    def __createInfoNode ( self, InfoStrings ):

        anInfoElement = self.__createElement( 'info' )
        anInfoData = self.__theDocument.createTextNode( InfoStrings )
        anInfoElement.appendChild( anInfoData )

        return anInfoElement
    


def convertSystemFullID2SystemID( aSystemFullID ):
    """
    aSystemFullID : ex) System:/CELL:CYTOPLASM
    return -> aSystemID [string] : ex) /CELL/CYTOPLASM
    """
    _dummy, aSystemPath, anID = aSystemFullID.split( ':' )

    if( anID == '/' ):
        return '/'

    elif( aSystemPath == '/' ):
        return '/' + anID

    else:
        return aSystemPath + '/' + anID


def convertSystemID2SystemFullID( aSystemID ):
    """
    aSystemID : ex) /CELL/CYTOPLASM
    return -> aSystemFullID [string] : ex) System:/CELL:CYTOPLASM
    """

    if ( aSystemID == '/' ):
        return 'System::/'

    aLastSlash = aSystemID.rfind( '/' )

    # subsystems of root: e.g. /CELL
    if aLastSlash == 0:
        return 'System:/:' + aSystemID[aLastSlash+1:]
    else: # others: e.g. /CELL/CYTOPLASM
        return 'System:' + aSystemID[:aLastSlash] + ':' +\
               aSystemID[aLastSlash+1:]

