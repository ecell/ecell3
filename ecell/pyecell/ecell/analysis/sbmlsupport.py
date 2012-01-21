#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2012 Keio University
#       Copyright (C) 2005-2009 The Molecular Sciences Institute
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
"""

__program__ = 'sbmlsupport'
__version__ = '1.0'
__author__ = 'Kazunari Kaizu <kaizu@sfc.keio.ac.jp>'
__copyright__ = ''
__license__ = ''


import re
import exceptions
import xml.dom.minidom

import libsbml

import ecell.ecssupport


ECELL_XML_NAMESPACE_PREFIX = 'ecell'
ECELL_XML_NAMESPACE_URI = 'http://www.sbml.org/2001/ns/ecell'

ODE_STEPPER_ID = 'DES01'
PASSIVE_STEPPER_ID = 'PS01'

ROOT_SYSTEM_ID = 'default'

AVOGADRO_CONSTANT = 6.0221367e+23


class SBMLConvertError( exceptions.Exception ):


    def __init__( self, messageString ):

        self.messageString = messageString

    # end of __init__


    def __str__( self ):

        return repr( self.messageString )

    # end of __str__


# end of SBMLConvertError


def createIdFromFullID( fullIDString ):

    fullID = ecell.ecssupport.createFullID( fullIDString )

    if fullID[ 2 ] == '/':
        return ROOT_SYSTEM_ID
    elif fullID[ 0 ] == ecell.ecssupport.VARIABLE and fullID[ 2 ] == 'SIZE':
        # WARNING: recursive call
        fullID = ecell.ecssupport.createFullIDFromSystemPath( fullID[ 1 ] )
        fullIDString = ecell.ecssupport.createFullIDString( fullID )
        return createIdFromFullID( fullIDString )
    else:
        return fullID[ 2 ]

# end of createIdFromFullID


def getSBaseAnnotation( sbaseObj, pn, aXMLNamespaceList ):

    propertyName = '%s:%s' % ( ECELL_XML_NAMESPACE_PREFIX, pn )

    if sbaseObj.isSetAnnotation():
        annotationString = '<xml'
        for i in range( aXMLNamespaceList.getLength() ):
            annotationString += ' xmlns:%s="%s"' % ( aXMLNamespaceList.getPrefix( i ), aXMLNamespaceList.getURI( i ) )
        annotationString = '%s>%s</xml>' % ( annotationString, sbaseObj.getAnnotation() )

        annotationDoc = xml.dom.minidom.parseString( annotationString )
        annotationNode = annotationDoc.childNodes[ 0 ].childNodes[ 0 ]

        targetNode = None
        for aNode in annotationNode.childNodes:
            if str( aNode.nodeName ) == propertyName:
                targetNode = aNode
                break

        if targetNode != None:
            if len( targetNode.childNodes ) != 1:
                raise SBMLConvertError, 'Annotation [%s] is invalid' % ( propertyName )
            return targetNode.childNodes[ 0 ].data

        annotationDoc.unlink()

    return None

# end of getSBaseAnnotation


def setSBaseAnnotation( sbaseObj, pn, value, aXMLNamespaceList ):

    if type( value ) != str:
        raise TypeError, 'Value must be a string'

    propertyName = '%s:%s' % ( ECELL_XML_NAMESPACE_PREFIX, pn )
    if not sbaseObj.isSetAnnotation():
        annotationString = '<annotation><%s>%s</%s></annotation>' % ( propertyName, value, propertyName )
        sbaseObj.setAnnotation( annotationString )

    else:
        annotationString = '<xml'
        for i in range( aXMLNamespaceList.getLength() ):
            annotationString += ' xmlns:%s="%s"' % ( aXMLNamespaceList.getPrefix( i ), aXMLNamespaceList.getURI( i ) )
        annotationString = '%s>%s</xml>' % ( annotationString, sbaseObj.getAnnotation() )

        annotationDoc = xml.dom.minidom.parseString( annotationString )
        annotationNode = annotationDoc.childNodes[ 0 ].childNodes[ 0 ]

        targetNode = None
        for aNode in annotationNode.childNodes:
            if str( aNode.nodeName ) == propertyName:
                targetNode = aNode
                break

        if not targetNode:
            targetNode = annotationDoc.createElement( propertyName )
            targetNode.appendChild( annotationDoc.createTextNode( value ) )
            annotationNode.appendChild( targetNode )
            sbaseObj.setAnnotation( str( annotationNode.toxml() ) )

        else:
            if len( targetNode.childNodes ) != 1:
                raise SBMLConvertError, 'Annotation [%s] is invalid' % ( propertyName )
            if targetNode.childNodes[ 0 ].data != value:
                ## print 'Annotation [%s] is overwritten' % ( propertyName )
                targetNode.childNodes[ 0 ].data = value

# end of setSBaseAnnotation


class SBMLIdManager:


    def __init__( self, aSBMLDocument ):

        self.idRegex = re.compile( '[a-zA-Z_][a-zA-Z0-9_]*' )

        self.namespace = 'System::/'

        self.theXMLNamespaceList = aSBMLDocument.getNamespaces()

        self.initialize( aSBMLDocument.getModel() )

    # end of __init__


    def initialize( self, aModel ):

        self.compartmentDict = {}
        self.speciesDict = {}
        self.parameterDict = {}
        self.reactionDict = {}

        self.ruleDict = {}

        self.idDict = None
        
        self.__checkIds( aModel )

    # end of initialize


    def getNamespace( self ):

        return self.namespace

    # end of self.namespace


    def isIdExist( self, id ):

        return ( self.compartmentDict.has_key( id ) \
                 or self.speciesDict.has_key( id ) \
                 or self.parameterDict.has_key( id ) \
                 or self.reactionDict.has_key( id ) )

    # end of isIdExist
    

    def getCompartmentFullID( self, id ):

        if self.compartmentDict.has_key( id ):
            return self.compartmentDict[ id ]
        else:
            return None

    # end of getCompartmentFullID
    
    
    def getSpeciesFullID( self, id ):
        
        if self.speciesDict.has_key( id ):
            return self.speciesDict[ id ]
        else:
            return None

    # end of getSpeciesFullID

    
    def getParameterFullID( self, id ):
        
        if self.parameterDict.has_key( id ):
            return self.parameterDict[ id ]
        else:
            return None

    # end of getParameterFullID


    def getReactionFullID( self, id ):
        
        if self.reactionDict.has_key( id ):
            return self.reactionDict[ id ]
        else:
            return None

    # end of getReactionFullID


    def getIdFromFullID( self, fullIDString ):
        '''
        idDict is redundant. Substitute searchIdFromFullID if you
        do not want to make idDict
        '''

        if not self.idDict:
            self.idDict = self.createIdDict()

        if self.idDict.has_key( fullIDString ):
            return self.idDict[ fullIDString ]
        else:
            return ( None, None )

    # end of getIdFromFullID
    

    def searchIdFromFullID( self, fullIDString ):

        fullID = ecell.ecssupport.createFullID( fullIDString )

        if fullID[ 0 ] == ecell.ecssupport.VARIABLE and fullID[ 2 ] == 'SIZE':
            fullID = ecell.ecssupport.createFullIDFromSystemPath( fullID[ 1 ] )
            fullIDString = ecell.ecssupport.createFullIDString( fullID )

        if fullID[ 0 ] == ecell.ecssupport.SYSTEM:
            for id in self.compartmentDict.keys():
                if self.compartmentDict[ id ] == fullIDString:
                    return ( id, libsbml.SBML_COMPARTMENT )

        elif fullID[ 0 ] == ecell.ecssupport.VARIABLE:
            for id in self.speciesDict.keys():
                if self.speciesDict[ id ] == fullIDString:
                    return ( id, libsbml.SBML_SPECIES )

            for id in self.parameterDict.keys():
                if self.parameterDict[ id ] == fullIDString:
                    return ( id, libsbml.SBML_PARAMETER )

        elif fullID[ 0 ] == ecell.ecssupport.PROCESS:
            for id in self.reactionDict.keys():
                if self.reactionDict[ id ] == fullIDString:
                    return ( id, libsbml.SBML_REACTION )

            for id in self.ruleDict.keys():
                if self.ruleDict[ id ][ 0 ] == fullIDString:
                    return ( id, self.ruleDict[ id ][ 1 ] )

        return ( None, None )

    # end of searchIdFromFullID
    

    def searchFullIDFromId( self, id ):

        fullIDString = self.getCompartmentFullID( id )
        if fullIDString != None:
            return ( fullIDString, libsbml.SBML_COMPARTMENT )

        fullIDString = self.getSpeciesFullID( id )
        if fullIDString != None:
            return ( fullIDString, libsbml.SBML_SPECIES )

        fullIDString = self.getParameterFullID( id )
        if fullIDString != None:
            return ( fullIDString, libsbml.SBML_PARAMETER )

##         fullIDString = self.getReactionFullID( id )
##         if fullIDString != None:
##             return ( fullIDString, libsbml.SBML_REACTION )

        return ( None, None )

    # end of searchFullIDFromId
    

    def getRuleFullID( self, i ):
        
        if self.ruleDict.has_key( i ):
            return self.ruleDict[ i ][ 0 ]
        else:
            return None

    # end of getRuleFullID


    def createIdDict( self ):

        if self.idDict != None:
            return self.idDict
            
        idDict = {}

        for id in self.compartmentDict.keys():
            fullIDString = self.compartmentDict[ id ]

            idDict[ fullIDString ] = ( id, libsbml.SBML_COMPARTMENT )

            fullID = ecell.ecssupport.createFullID( fullIDString )
            systemPath = ecell.ecssupport.createSystemPathFromFullID( fullID )
            idDict[ 'Variable:%s:SIZE' % ( systemPath ) ] \
                    = ( id, libsbml.SBML_COMPARTMENT )
            
        for id in self.speciesDict.keys():
            idDict[ self.speciesDict[ id ] ] = ( id, libsbml.SBML_SPECIES )

        for id in self.parameterDict.keys():
            idDict[ self.parameterDict[ id ] ] = ( id, libsbml.SBML_PARAMETER )

        for id in self.reactionDict.keys():
            idDict[ self.reactionDict[ id ] ] = ( id, libsbml.SBML_REACTION )

        for id in self.ruleDict.keys():
            ( fullIDString, ruleType ) = self.ruleDict[ id ]
            idDict[ fullIDString ] = ( id, ruleType )

        return idDict

    # end of createIdDict
    
    
    def __checkIds( self, aModel ):

        ## __createCompartmentFullID must be called primarily
        fullIDList = []
        for i in range( aModel.getNumCompartments() ):
            aCompartment = aModel.getCompartment( i )
            id = aCompartment.getId()
            if not id:
                raise SBMLConvertError, 'Compartment [%s] has no Id' % ( i )

            fullIDString = self.__createCompartmentFullID( aModel, id )
            if fullIDList.count( fullIDString ) != 0:
                raise SBMLConvertError, \
                      'FullID [%s] is assigned twice' % ( fullIDString )
            fullIDList.append( fullIDString )

            self.compartmentDict[ id ] = fullIDString

        fullIDList = []
        for i in range( aModel.getNumSpecies() ):
            aSpecies = aModel.getSpecies( i )
            id = aSpecies.getId()
            if not id:
                raise SBMLConvertError, 'Species [%s] has no Id' % ( i )

            fullIDString = self.__createSpeciesFullID( aModel, id )
            if fullIDList.count( fullIDString ) != 0:
                raise SBMLConvertError, \
                      'FullID [%s] is assigned twice' % ( fullIDString )
            fullIDList.append( fullIDString )

            self.speciesDict[ id ] = fullIDString

        for i in range( aModel.getNumParameters() ):
            aParameter = aModel.getParameter( i )
            id = aParameter.getId()
            if not id:
                raise SBMLConvertError, 'Parameter [%s] has no Id' % ( i )

            fullIDString = self.__createParameterFullID( aModel, id )
            if fullIDList.count( fullIDString ) != 0:
                raise SBMLConvertError, \
                      'FullID [%s] is assigned twice' % ( fullIDString )
            fullIDList.append( fullIDString )

            self.parameterDict[ id ] = fullIDString

        fullIDList = []
        for i in range( aModel.getNumReactions() ):
            aReaction = aModel.getReaction( i )
            id = aReaction.getId()
            if not id:
                raise SBMLConvertError, 'Reaction [%s] has no Id' % ( i )

            fullIDString = self.__createReactionFullID( aModel, id )
            if fullIDList.count( fullIDString ) != 0:
                raise SBMLConvertError, \
                      'FullID [%s] is assigned twice' % ( fullIDString )
            fullIDList.append( fullIDString )

            self.reactionDict[ id ] = fullIDString

        for i in range( aModel.getNumRules() ):
            aRule = aModel.getRule( i )

            fullIDString = self.__createRuleFullID( aModel, i )
            if fullIDList.count( fullIDString ) != 0:
                raise SBMLConvertError, \
                      'FullID [%s] is assigned twice' % ( fullIDString )
            fullIDList.append( fullIDString )

            self.ruleDict[ i ] = ( fullIDString, aRule.getTypeCode() )

    # end of checkIds
    

    def __createCompartmentFullID( self, aModel, sbmlId ):

        aCompartment = aModel.getCompartment( sbmlId )
        
        annotation = getSBaseAnnotation( aCompartment, 'ID', self.theXMLNamespaceList )
        if annotation == '/':
            return self.getNamespace()

        if not aCompartment.isSetOutside():
            fullIDString = self.getNamespace()
        else:
            outsideId = aCompartment.getOutside()
            fullIDString = self.__createCompartmentFullID( aModel, outsideId )

        fullID = ecell.ecssupport.createFullID( fullIDString )
        systemPath = ecell.ecssupport.createSystemPathFromFullID( fullID )

        return 'System:%s:%s' % ( systemPath, self.__createSObjectID( aCompartment, self.theXMLNamespaceList ) )

    # end of __createCompartmentFullID


    def __createSpeciesFullID( self, aModel, sbmlId ):

        aSpecies = aModel.getSpecies( sbmlId )

        if not aSpecies.isSetCompartment():
            raise SBMLConvertError, 'Species [%s] has no compartment' % ( sbmlId )

        fullIDString = self.getCompartmentFullID( aSpecies.getCompartment() )
        if not fullIDString:
            raise SBMLConvertError, 'Compartment [%s] is not found' % ( aSpecies.getCompartment() )
        
        fullID = ecell.ecssupport.createFullID( fullIDString )
        systemPath = ecell.ecssupport.createSystemPathFromFullID( fullID )

        return 'Variable:%s:%s' % ( systemPath, self.__createSObjectID( aSpecies, self.theXMLNamespaceList ) )

    # end of __createSpeciesFullID
    

    def __createParameterFullID( self, aModel, sbmlId ):

        aParameter = aModel.getParameter( sbmlId )

        annotation = getSBaseAnnotation( aParameter, 'compartment', self.theXMLNamespaceList )
        if annotation != None and self.idRegex.match( annotation ):
            fullIDString = self.getCompartmentFullID( annotation )
            if not fullIDString:
                raise SBMLConvertError, 'Compartment [%s] is not found' % ( annotation )
        
        else:
            fullIDString = self.getNamespace()

        fullID = ecell.ecssupport.createFullID( fullIDString )
        systemPath = ecell.ecssupport.createSystemPathFromFullID( fullID )
            
        return 'Variable:%s:%s' % ( systemPath, self.__createSObjectID( aParameter, self.theXMLNamespaceList ) )

    # end of __createParameterFullID
    

    def __createReactionFullID( self, aModel, sbmlId ):

        aReaction = aModel.getReaction( sbmlId )

        annotation = getSBaseAnnotation( aReaction, 'compartment', self.theXMLNamespaceList )
        if annotation != None and self.idRegex.match( annotation ):
            fullIDString = self.getCompartmentFullID( annotation )
            if not fullIDString:
                raise SBMLConvertError, 'Compartment [%s] is not found' % ( annotation )
        else:
            fullIDString = self.getNamespace()

        fullID = ecell.ecssupport.createFullID( fullIDString )
        systemPath = ecell.ecssupport.createSystemPathFromFullID( fullID )
            
        return 'Process:%s:%s' % ( systemPath, self.__createSObjectID( aReaction, self.theXMLNamespaceList ) )

    # end of __createReactionFullID


    def __createRuleFullID( self, aModel, i ):

        aRule = aModel.getRule( i )
            
        annotation = getSBaseAnnotation( aRule, 'compartment', self.theXMLNamespaceList )
        if annotation != None and self.idRegex.match( annotation ):
            fullIDString = self.getCompartmentFullID( annotation )
            if not fullIDString:
                raise SBMLConvertError, 'Compartment [%s] is not found' % ( annotation )
        else:
            fullIDString = self.getNamespace()

        fullID = ecell.ecssupport.createFullID( fullIDString )
        systemPath = ecell.ecssupport.createSystemPathFromFullID( fullID )
        
        return 'Process:%s:%s' % ( systemPath, self.__createSObjectID( aRule, self.theXMLNamespaceList, 'Rule%s' % ( i ) ) )
    
    # end of __createRuleFulllID


    def __createSObjectID( self, sbmlObj, aXMLNamespaceList, default=None ):

        if not default:
            default = sbmlObj.getId()
        
        annotation = getSBaseAnnotation( sbmlObj, 'ID', self.theXMLNamespaceList )
        if annotation != None and self.idRegex.match( annotation ):
            return annotation
        else:
            return default

    # end of __createSObjectID


# end of SBMLIdManager


if __name__ == '__main__':
    
    import sbmlsupport
    import sys
    import os.path


    def main( filename ):

        sbmlDocument = libsbml.readSBML( filename )
        aSBMLIdManager = sbmlsupport.SBMLIdManager( sbmlDocument )

        print aSBMLIdManager.compartmentDict
        print aSBMLIdManager.speciesDict
        print aSBMLIdManager.parameterDict
        print aSBMLIdManager.reactionDict
        print aSBMLIdManager.ruleDict

    # end of main


    if len( sys.argv ) > 1:
        main( sys.argv[ 1 ] )
