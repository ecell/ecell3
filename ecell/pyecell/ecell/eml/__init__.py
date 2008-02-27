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

from xml.dom import minidom

import string
import types

from types import *
from ecell.ecssupport import *

class Eml:
    def __init__( self, aFileObject = None ):
        """read EML file and make domtree"""

        if aFileObject is None:
            aStringData = '<?xml version="1.0" ?><eml></eml>'
        elif type( aFileObject ) == str or type( aFileObject ) == unicode:
            aStringData = aFileObject
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
        

        self.__reconstructCache()

    def asString( self ):
        """return domtree as string"""

        return self.__theDocument.toprettyxml(indent="", newl="\n")

    def save( self, anOutputFile ):
        """save domtree as an EML file"""
        
        anEmlString = self.asString()
        
        anOutputFileObject = open( anOutputFile, 'w' )
        anOutputFileObject.write( anEmlString )

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
            
    def createEntity( self, aClass, aFullID ):
        anEntityElement = self.__createElement( aFullID.getTypeName().lower() )
        anEntityElement.setAttribute( 'class', aClass )

        if aFullID.typeCode == SYSTEM:
            if aTargetPath != '':  # check if the supersystem exists
                dummy = self.__getSystemNode( aFullID.getSuperSystemPath() )

            anID = convertSystemFullID2SystemID( aFullID )
            anEntityElement.setAttribute( 'id', aFullID.id )
            self.__theDocument.documentElement.appendChild( anEntityElement )

        elif aFullID.typeCode == VARIABLE or aFullID.typeCode == PROCESS:
            anEntityElement.setAttribute( 'id', aFullID.id )
            aTargetSystemNode = self.__getSystemNode( aFullID.getSuperSystemPath() )
            aTargetSystemNode.appendChild( anEntityElement )

        else:            
            raise "unexpected error. %s should be System, Variable, or Process."%anEntityType

        self.__addToCache( aFullID, anEntityElement )

    def deleteEntity( self, aFullID ):
        """delete an entity"""
        if aFullID.typeCode == SYSTEM:
            for anElement in self.__theEmlNode.childNodes:
                if SystemPath( anElement.getAttribute( 'id' ) ).toFullID() == aFullID:
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
                       aSystemNode.getAttribute( 'id' ) == str( aSystemPath ):
                    for aChildNode in aSystemNode.childNodes:
                        if aChildNode.nodeName == aType:
                            anEntityList.append( str( aChildNode.getAttribute( 'id' ) ) )

        return anEntityList

    def getEntityPropertyMap( self, aFullID ):
        anEntityNode = self.__getEntityNode( aFullID )
        aRetval = {}
        
        for aChildNode in anEntityNode.childNodes:
            if aChildNode.nodeName == 'property':
                aRetval[ aChildNode.getAttribute( 'name' ) ] =  \
                    self.__createValueList( aChildNode )
        return aRetval

    def getEntityPropertyList( self, aFullID ):
        anEntityNode = self.__getEntityNode( aFullID )
        aRetval = []
        
        for aChildNode in anEntityNode.childNodes:
            if aChildNode.nodeName == 'property':
                aRetval.append(
                    str( aChildNode.getAttribute( 'name' ) ) )

        return aRetval

    def getEntityProperty( self, aFullPN ):
        assert isinstance( aFullPN, identifiers.FullPN )
        anEntityPropertyNode = self.__getEntityPropertyNode(
            aFullPN.fullID, aFullPN.propertyName )
        return self.__createValueList( anEntityPropertyNode )

    def setEntityInfo( self, aFullID, InfoStrings ):
        anEntityInfoElement = self.__createInfoNode( InfoStrings )
        aTargetNode = self.__getEntityNode( aFullID )
        aTargetNode.appendChild( anEntityInfoElement )

    def getEntityInfo( self, aFullID ):
        anEntityInfoNode = self.__getEntityInfoNode( aFullID )

        return self.__createValueList( anEntityInfoNode )
    
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
                aSystemFullID = identifiers.SystemPath( aSystemPath )
                self.__addToCache( aSystemFullID, aSystemNode )

                for aChildNode in aSystemNode.childNodes:
                    aType = string.capwords( aChildNode.nodeName )

                    if  aType in ( 'Variable', 'Process' ):
                        anID = aChildNode.getAttribute( 'id' )
                        aFullID = aType + ':' + aSystemPath + ':' + anID
                        self.__addToCache( aFullID, aChildNode )

    def __createValueList( self, aContainerNode ):
        retval = []
        for aNode in aContainerNode.childNodes:
            aNodeType = aNode.nodeType
            if aNodeType == aNode.TEXT_NODE:
                aValue = aNode.nodeValue.strip()
                if len( aValue ) == 0:
                    continue
                aValue = aValue.replace( '#x0A', '\n')
                retval.append( aValue )
            elif aNodeType == aNode.ELEMENT_NODE and aNode.nodeName == 'value':
                retval.append( self.__createValueList( aNode ) )
            else:
                raise "unexpected error."
        if len( retval ) == 1 and type( retval[ 0 ] ) != list:
            retval = retval[ 0 ]
        return retval

    def __getSubSystemList( self, aSystemPath ):
        aSystemList = []
        # if '' is given, return the root system ('/')
        if len( aSystemPath ) == 1:
            for aSystemNode in self.__theEmlNode.childNodes:
                if aSystemNode.nodeName == 'system' and \
                     aSystemNode.getAttribute( 'id' ) == '/':
                    return [ '/' ]
        else:
            for aSystemNode in self.__theEmlNode.childNodes:
                if aSystemNode.nodeName == 'system':
                    aCandidate = identifiers.SystemPath(
                        aSystemNode.getAttribute( 'id' ) )
                    if aSystemPath.isParentOf( aCandidate ):
                        aSystemList.append( aCandidate.toFullID().id )

        return aSystemList

    def __getEntityNode( self, aFullID ):
        # first look up the cache
        try:
            return self.__findInCache( aFullID )
        except:
            pass


        if aFullID.typeCode == SYSTEM:
            return self.__getSystemNode( aFullID.toSystemPath() )
            
        aSystemNode = self.__getSystemNode( aFullID.getSuperSystemPath() )

        for aChildNode in aSystemNode.childNodes:
            if aChildNode.nodeName.upper() == aFullID.getTypeName().upper() and\
                   aChildNode.getAttribute( 'id' ) == aFullID.id:
                self.__addToCache( aFullID, aChildNode )
                return aChildNode

        raise "Entity [%s] not found."%aFullID
                        
    def __getSystemNode( self, aSystemPath ):
        aFullID = identifiers.SystemPath( aSystemPath )

        # first look up the cache
        try:
            return self.__findInCache( aFullID )
        except:
            pass

        for aSystemNode in self.__theEmlNode.childNodes:
            if aSystemNode.nodeName == 'system' and \
                   str( aSystemNode.getAttribute( 'id' ) ) == str( aSystemPath ):
                self.__addToCache( aFullID, aSystemNode )
                return aSystemNode
        raise RuntimeError( "System [%s] not found." % aFullID )

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

        if type( aValue ) in ( types.TupleType, types.ListType ):
            # vector value
            map( aValueNode.appendChild,\
                 map( self.__createValueNode, aValue ) )

        else:
            # scaler value
            aNormalizedValue =  string.replace( aValue, '\n', '#x0A' )

            aValueData = self.__theDocument.createTextNode( aNormalizedValue )
            aValueNode.appendChild( aValueData )


        return aValueNode

    def __createInfoNode ( self, InfoStrings ):
        anInfoElement = self.__createElement( 'info' )
        anInfoData = self.__theDocument.createTextNode( InfoStrings )
        anInfoElement.appendChild( anInfoData )

        return anInfoElement

