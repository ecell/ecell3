#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2014 Keio University
#       Copyright (C) 2008-2014 RIKEN
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

__program__ = 'SBMLExporter'
__version__ = '1.0'
__author__ = 'Kazunari Kaizu <kaizu@sfc.keio.ac.jp>'
__copyright__ = ''
__license__ = ''


import re
from xml.dom import minidom

import libsbml

import ecell.eml
import ecell.ecssupport
import ecell.expressionparser

import ecell.analysis.emlsupport
import ecell.analysis.util

from sbmlsupport import *


class ExpressionParser:


    def __init__( self, namespaceDict, compartment=None ):

        self.namespaceDict = namespaceDict
        self.modifierList = []
        self.compartment = compartment

    # end of __init__


    def parse( self, expressionString ):

        self.modifierList = []

        regexpr = re.compile( '([a-zA-Z_][a-zA-Z0-9_]*.Value\s*\/\s*[a-zA-Z_][a-zA-Z0-9_]*.Value)' )
        stripexpr = re.compile( '(.Value)|\s' )

        expressionList = regexpr.split( expressionString )
        for c in range( 1, len( expressionList ), 2 ):
            ( bunshiName, bunboName ) \
              = stripexpr.sub( '', expressionList[ c ] ).split( '/' )

            if self.namespaceDict.has_key( bunshiName ) \
                   and self.namespaceDict.has_key( bunboName ):
                
                ( sbmlId, coeff ) = self.namespaceDict[ bunshiName ]
                if type( sbmlId ) == tuple and sbmlId[ 1 ] == bunboName:
                    expressionList[ c ] = sbmlId[ 0 ]

                    if coeff == 0 \
                           and self.modifierList.count( sbmlId[ 0 ] ) == 0:
                        self.modifierList.append( sbmlId[ 0 ] )

        formulaString = ''.join( expressionList )
            
        regexpr = re.compile( '([a-zA-Z_][a-zA-Z0-9_]*.Value)' )
        expressionList = regexpr.split( formulaString )
        for c in range( 1, len( expressionList ), 2 ):
            referenceName = expressionList[ c ].replace( '.Value', '' )
            if self.namespaceDict.has_key( referenceName ):
                ( sbmlId, coeff ) = self.namespaceDict[ referenceName ]
                if type( sbmlId ) == str:
                    expressionList[ c ] = sbmlId
                elif type( sbmlId ) == tuple:
                    expressionList[ c ] = '%s * %s' % sbmlId
                    if coeff == 0 \
                           and self.modifierList.count( sbmlId[ 0 ] ) == 0:
                        self.modifierList.append( sbmlId[ 0 ] )
                        
        formulaString = ''.join( expressionList )

        regexpr = re.compile( '([a-zA-Z_][a-zA-Z0-9_]*.MolarConc)' )
        expressionList = regexpr.split( formulaString )
        for c in range( 1, len( expressionList ), 2 ):
            referenceName = expressionList[ c ].replace( '.MolarConc', '' )
            if self.namespaceDict.has_key( referenceName ):
                ( sbmlId, coeff ) = self.namespaceDict[ referenceName ]
                if type( sbmlId ) == str:
                    raise SBMLConvertError, \
                          'MolarConc of [%s] is not supported' \
                          % ( referenceName )
                elif type( sbmlId ) == tuple:
                    expressionList[ c ] = '%s / %e' % ( sbmlId[ 0 ], AVOGADRO_CONSTANT )
                    if coeff == 0 \
                           and self.modifierList.count( sbmlId[ 0 ] ) == 0:
                        self.modifierList.append( sbmlId[ 0 ] )         

        formulaString = ''.join( expressionList )
        formulaString = formulaString.replace( 'self.getSuperSystem().SizeN_A', '( %s * %e )' % ( self.compartment, AVOGADRO_CONSTANT ) )

        regexpr = re.compile( '([a-zA-Z_][a-zA-Z0-9_]*.Value\s*\/\s*[a-zA-Z_][a-zA-Z0-9_]*.Value)' )

        if regexpr.match( formulaString ) != None:
            raise SBMLConvertError, \
                  'Expression [%s] cannot be coverted successfully' \
                  % ( expressionString )
        
        return formulaString

    # end of parse
    

    def convertExpressionToASTNode( self, expressionString ):

        anExpression = self.parse( expressionString )

        anASTNode = libsbml.parseFormula( anExpression )
        if not anASTNode:
            raise SBMLConvertError, \
                  'Expression [%s] cannot be coverted successfully' \
                  % ( anExpression )

        return anASTNode

    # end of convertExpressionToASTNode
    

# end of ExpressionParser


class SBMLExporter:


    def __init__( self, filename ):

        self.initialize( filename )

    # end of __init__


    def initialize( self, filename ):

        self.theEml = ecell.analysis.emlsupport.EmlSupport( filename )

        self.theSBMLDocument = None

    # end of initialize
    
    
    def loadSBML( self, aSBMLDocument ):

        if type( aSBMLDocument ) == str:
            self.theSBMLDocument = libsbml.readSBML( aSBMLDocument )
        elif type( self.theSBMLDocument ) == file:
            self.theSBMLDocument = libsbml.readSBMLFromString( aSBMLDocument.read() )
        else:
            self.theSBMLDocument = aSBMLDocument
            
        if self.theSBMLDocument.getLevel() != 2:
            self.theSBMLDocument.setLevel( 2 )
            
        aXMLNamespaceList = self.theSBMLDocument.getNamespaces()
        anURI = aXMLNamespaceList.getURI( ECELL_XML_NAMESPACE_PREFIX )
        if anURI != ECELL_XML_NAMESPACE_URI:
            if anURI == '':
                aXMLNamespaceList.add( ECELL_XML_NAMESPACE_PREFIX, \
                                       ECELL_XML_NAMESPACE_URI )
            else:
                raise SBMLConvertError, 'Wrong URI [%s] is already assinged to xmlns:%s. Remove or correct it by yourself' % ( anURI, ECELL_XML_NAMESPACE_PREFIX )

        if self.theSBMLDocument.getNumFatals() > 0:
            fatalerrorList = []
            for i in range( self.theSBMLDocument.getNumFatals() ):
                fatalerrorList.append( self.theSBMLDocument.getFatal( i ).getMessage() )
                
            self.theSBMLDocument = None
            raise SBMLConvertError, \
                  'This SBML document is invalid:\n %s' \
                  % ( ', '.join( fatalerrorList ) )

        if not self.theSBMLDocument.getModel():
            self.theSBMLDocument.createModel()

        self.__createIdDict()
        
    # end of loadSBML
    

    def convertEMLToSBML( self ):

        if not self.theSBMLDocument:
            self.loadSBML( libsbml.SBMLDocument() )

        for fullIDString in self.theEml.getSystemList():
            ( id, sbaseType ) = self.searchIdFromFullID( fullIDString )
            aCompartmentExporter = CompartmentExporter( self, fullIDString, id )
            aCompartmentExporter.writeSBML( self.theSBMLDocument )

        for fullIDString in self.theEml.getVariableList():
            ( id, sbaseType ) = self.searchIdFromFullID( fullIDString )

            if sbaseType == libsbml.SBML_SPECIES:
                aSpeciesExporter = SpeciesExporter( self, fullIDString, id )
                aSpeciesExporter.writeSBML( self.theSBMLDocument )

            elif sbaseType == libsbml.SBML_PARAMETER:
                aParameterExporter = ParameterExporter( self, fullIDString, id )
                aParameterExporter.writeSBML( self.theSBMLDocument )

        for fullIDString in self.theEml.getProcessList():
            ( id, sbaseType ) = self.searchIdFromFullID( fullIDString )

            if sbaseType == libsbml.SBML_REACTION:
                aReactionExporter = ReactionExporter( self, fullIDString, id )
                aReactionExporter.writeSBML( self.theSBMLDocument )

            elif sbaseType == libsbml.SBML_ASSIGNMENT_RULE:
                aRuleExporter = RuleExporter( self, fullIDString, id )
                aRuleExporter.writeSBML( self.theSBMLDocument )

        return self.theSBMLDocument
        
    # end of convertEMLToSBML


    def saveAsSBML( self, filename ):

        aSBMLDocument = self.convertEMLToSBML()
        libsbml.writeSBML( aSBMLDocument, filename )

    # end of saveAsSBML


    def __createIdDict( self ):

        if not self.theSBMLDocument:
            return

        aSBMLIdManager = SBMLIdManager( self.theSBMLDocument )
        self.idDict = aSBMLIdManager.createIdDict()
        del aSBMLIdManager

        namingDict = {}

        for fullIDString in self.theEml.getSystemList():
            if not self.idDict.has_key( fullIDString ):
                id = createIdFromFullID( fullIDString )
                if namingDict.has_key( id ):
                    ( targetFullIDString, sbmlType ) = namingDict.pop( id )
                    name = ecell.analysis.util.convertToDataString( targetFullIDString )
                    namingDict[ name ] = ( targetFullIDString, sbmlType )

                    name = ecell.analysis.util.convertToDataString( fullIDString )
                    namingDict[ name ] \
                                = ( fullIDString, libsbml.SBML_COMPARTMENT )
                else:
                    namingDict[ id ] \
                                = ( fullIDString, libsbml.SBML_COMPARTMENT )

        for fullIDString in self.theEml.getVariableList():
            fullID = ecell.ecssupport.createFullID( fullIDString )
            if fullID[ 0 ] == ecell.ecssupport.VARIABLE \
                   and fullID[ 2 ] == 'SIZE':
                pass
                
            elif not self.idDict.has_key( fullIDString ):
                id = createIdFromFullID( fullIDString )
                if namingDict.has_key( id ):
                    ( targetFullIDString, sbmlType ) = namingDict.pop( id )
                    name = ecell.analysis.util.convertToDataString( targetFullIDString )
                    namingDict[ name ] = ( targetFullIDString, sbmlType )

                    name = ecell.analysis.util.convertToDataString( fullIDString )
                    namingDict[ name ] \
                                = ( fullIDString, libsbml.SBML_SPECIES )
                else:
                    namingDict[ id ] \
                                = ( fullIDString, libsbml.SBML_SPECIES )

        for fullIDString in self.theEml.getProcessList():
            if not self.idDict.has_key( fullIDString ):

                className = self.theEml.getEntityClass( fullIDString ) 
                if className == 'ExpressionFluxProcess':
                    id = createIdFromFullID( fullIDString )
                    if namingDict.has_key( id ):
                        ( targetFullIDString, sbmlType ) = namingDict.pop( id )
                        name = ecell.analysis.util.convertToDataString( targetFullIDString )
                        namingDict[ name ] = ( targetFullIDString, sbmlType )
                        
                        name = ecell.analysis.util.convertToDataString( fullIDString )
                        namingDict[ name ] = ( fullIDString, \
                                               libsbml.SBML_REACTION )
                    else:
                        namingDict[ id ] = ( fullIDString, \
                                             libsbml.SBML_REACTION )

                elif className == 'ExpressionAssignmentProcess':
                    self.idDict[ fullIDString ] \
                                 = ( None, libsbml.SBML_ASSIGNMENT_RULE )

##                 else:
##                      raise SBMLConvertError, 'Class [%s] is not supported. Substitute  ExpressionFluxProcess or ExpressionAssignmentProcess' % ( className )

        for id in namingDict.keys():
            ( fullIDString, sbmlType ) = namingDict[ id ]
            self.idDict[ fullIDString ] = ( id, sbmlType )

            fullID = ecell.ecssupport.createFullID( fullIDString )
            if fullID[ 0 ] == ecell.ecssupport.SYSTEM:
                systemPath = ecell.ecssupport.createSystemPathFromFullID( fullID )
                self.idDict[ 'Variable:%s:SIZE' % ( systemPath ) ] \
                             = ( id, sbmlType )

    # end of __createIdDict
    
    
    def searchIdFromFullID( self, fullIDString ):

        if self.idDict.has_key( fullIDString ):
            return self.idDict[ fullIDString ]
        else:
            return ( None, None )
        
    # end of searchIdFromFullID
    

# end of SBMLExporter


class SBaseExporter:


    def __init__( self, aSBMLExporter, fullIDString, id=None ):

        self.initialize( aSBMLExporter, fullIDString, id )

    # end of __init__


    def initialize( self, aSBMLExporter, fullIDString, id=None ):

        self.theSBMLExporter = aSBMLExporter
        self.setFullID( fullIDString )

        if not id:
            # this id may be already assigned for different SBase
            self.setId( createIdFromFullID( fullIDString ) )
        else:
            self.setId( id )

    # end of initialize
    

    def setFullID( self, fullIDString ):

        if not self.theSBMLExporter.theEml.isEntityExist( fullIDString ):
            raise SBMLConvertError, \
                  'Entity [%s] is not found' % ( fullIDString )

        self.fullID = fullIDString

    # end of setFullID
    

    def setId( self, id=None ):

        self.id = id

    # end of setId


    def isSetId( self ):

        return ( ( createIdFromFullID( self.fullID ) != self.id )
                 or ( self.id == ROOT_SYSTEM_ID ) )
    
    # end of isSetId
    

    def writeSBML( self, aSBMLDocument ):

        import inspect
        caller = inspect.getouterframes( inspect.currentframe() )[ 0 ][ 3 ]
        raise NotImplementedError( '%s must be implemented in sub class' % ( caller ) )

    # end of writeSBML


# end of SBaseExporter


class CompartmentExporter( SBaseExporter ):


    def __init__( self, aSBMLExporter, fullIDString, id=None ):

        SBaseExporter.__init__( self, aSBMLExporter, fullIDString, id )

    # end of __init__


    def setFullID( self, fullIDString ):

        fullID = ecell.ecssupport.createFullID( fullIDString )
        if fullID[ 0 ] != ecell.ecssupport.SYSTEM:
            raise SBMLConvertError, \
                  'Entity [%s] is not System' % ( fullIDString )

        SBaseExporter.setFullID( self, fullIDString )

    # end of setFullID
    

    def writeSBML( self, aSBMLDocument ):

        aModel = aSBMLDocument.getModel()
        aCompartment = aModel.getCompartment( self.id )
        if not aCompartment:
            aCompartment = aModel.createCompartment()
            aCompartment.setId( self.id )

        fullID = ecell.ecssupport.createFullID( self.fullID )

        if self.isSetId():
            setSBaseAnnotation( aCompartment, 'ID', fullID[ 2 ], aSBMLDocument.getNamespaces() )

        outsideFullID = None
        if fullID[ 1 ] == '':
            if aCompartment.isSetOutside():
                aCompartment.unsetOutside()
        else:
            outsideFullID = ecell.ecssupport.createFullIDFromSystemPath( fullID[ 1 ] )
            outside = ecell.ecssupport.createFullIDString( outsideFullID )
            ( outside, sbmlType ) \
              = self.theSBMLExporter.searchIdFromFullID( outside )
            aCompartment.setOutside( outside )

        systemPath = ecell.ecssupport.createSystemPathFromFullID( fullID )
        sizeFullIDString = 'Variable:%s:SIZE' % ( systemPath )

        systemSize = 1.0

        if self.theSBMLExporter.theEml.isEntityExist( sizeFullIDString ):
            if self.theSBMLExporter.theEml.getEntityPropertyList( sizeFullIDString ).count( 'Value' ) == 1:
                systemSize = self.theSBMLExporter.theEml.getEntityProperty( '%s:Value' % ( sizeFullIDString ) )
                if len( systemSize ) != 1:
                    raise SBMLConvertError, \
                          'The format of property [%s:Value] is invalid' \
                          % ( sizeFullIDString )
##             else:
##                 raise SBMLConvertError, \
##                       'Variable [%s] has no value' % ( sizeFullIDString )

                systemSize = float( systemSize[ 0 ] )

        else:
            systemFullID = ecell.ecssupport.createFullID( self.fullID )
            while systemFullID[ 1 ] != '':
                outsideFullID = ecell.ecssupport.createFullIDFromSystemPath( systemFullID[ 1 ] )
                outsidePath = ecell.ecssupport.createSystemPathFromFullID( outsideFullID )
                outsideSizeFullIDString = 'Variable:%s:SIZE' % ( outsidePath )

                if self.theSBMLExporter.theEml.isEntityExist( outsideSizeFullIDString ) and self.theSBMLExporter.theEml.getEntityPropertyList( outsideSizeFullIDString ).count( 'Value' ) == 1:
                    outsideSize = self.theSBMLExporter.theEml.getEntityProperty( '%s:Value' % ( outsideSizeFullIDString ) )
                    if len( outsideSize ) != 1:
                        raise SBMLConvertError, \
                              'The format of property [%s:Value] is invalid' \
                              % ( outsideSizeFullIDString )
##                     else:
##                         raise SBMLConvertError, \
##                               'Variable [%s] has no value' \
##                               % ( outsideSizeFullIDString )

                    outsideSize = float( outsideSize[ 0 ] )
                    systemSize = outsideSize
                    break

                systemFullID = outsideFullID

        aCompartment.setSize( systemSize )

        propertyList = self.theSBMLExporter.theEml.getEntityPropertyList( self.fullID )
        for pn in propertyList:
            
            if pn == 'Name':
                name = self.theSBMLExporter.theEml.getEntityProperty( '%s:Name' % ( self.fullID ) )
                if len( name ) != 1:
                    raise SBMLConvertError, \
                          'The format of property [%s:Name] is invalid' \
                          % ( self.fullID )

                name = name[ 0 ]
                aCompartment.setName( name )

            elif pn == 'StepperID':
                pass
            
            else:
                raise SBMLConvertError, \
                      'Property [%s:%s] cannot be converted' \
                      % ( self.fullID, pn )

    # end of writeSBML


# end of CompartmentExporter


class SpeciesExporter( SBaseExporter ):


    def __init__( self, aSBMLExporter, fullIDString, id=None ):

        SBaseExporter.__init__( self, aSBMLExporter, fullIDString, id )

    # end of __init__


    def setFullID( self, fullIDString ):

        fullID = ecell.ecssupport.createFullID( fullIDString )
        if fullID[ 0 ] != ecell.ecssupport.VARIABLE:
            raise SBMLConvertError, \
                  'Entity [%s] is not Variable' % ( fullIDString )

        SBaseExporter.setFullID( self, fullIDString )

    # end of setFullID
    

    def writeSBML( self, aSBMLDocument ):

        aModel = aSBMLDocument.getModel()
        aSpecies = aModel.getSpecies( self.id )
        if not aSpecies:
            aSpecies = aModel.createSpecies()
            aSpecies.setId( self.id )

        fullID = ecell.ecssupport.createFullID( self.fullID )

        if self.isSetId():
            setSBaseAnnotation( aSpecies, 'ID', fullID[ 2 ], aSBMLDocument.getNamespaces() )

        compartment = ecell.ecssupport.createFullIDFromSystemPath( fullID[ 1 ] )
        compartment = ecell.ecssupport.createFullIDString( compartment )
        ( compartment, sbmlType ) \
          = self.theSBMLExporter.searchIdFromFullID( compartment )
        aSpecies.setCompartment( compartment )

        propertyList = self.theSBMLExporter.theEml.getEntityPropertyList( self.fullID )
        for pn in propertyList:
            
            if pn == 'Name':
                name = self.theSBMLExporter.theEml.getEntityProperty( '%s:Name' % ( self.fullID ) )
                if len( name ) != 1:
                    raise SBMLConvertError, \
                          'The format of property [%s:Name] is invalid' \
                          % ( self.fullID )

                name = name[ 0 ]
                aSpecies.setName( name )

            elif pn == 'Value':
                value = self.theSBMLExporter.theEml.getEntityProperty( '%s:Value' % ( self.fullID ) )
                if len( value ) != 1:
                    raise SBMLConvertError, \
                          'The format of property [%s:Value] is invalid' \
                          % ( self.fullID )

                value = float( value[ 0 ] )
                aSpecies.setInitialAmount( value )

            elif pn == 'Fixed':
                value = self.theSBMLExporter.theEml.getEntityProperty( '%s:Fixed' % ( self.fullID ) )
                if len( value ) != 1:
                    raise SBMLConvertError, \
                          'The format of property [%s:Fixed] is invalid' \
                          % ( self.fullID )

                if value[ 0 ] == '1':
                    aSpecies.setBoundaryCondition( True )

            else:
                raise SBMLConvertError, \
                      'Property [%s:%s] cannot be converted' \
                      % ( self.fullID, pn )

    # end of writeSBML


# end of SpeciesExporter


class ParameterExporter( SBaseExporter ):


    def __init__( self, aSBMLExporter, fullIDString, id=None ):

        SBaseExporter.__init__( self, aSBMLExporter, fullIDString, id )

    # end of __init__


    def setFullID( self, fullIDString ):

        fullID = ecell.ecssupport.createFullID( fullIDString )
        if fullID[ 0 ] != ecell.ecssupport.VARIABLE:
            raise SBMLConvertError, \
                  'Entity [%s] is not Variable' % ( fullIDString )

        SBaseExporter.setFullID( self, fullIDString )

    # end of setFullID
    

    def writeSBML( self, aSBMLDocument ):

        aModel = aSBMLDocument.getModel()
        aParameter = aModel.getParameter( self.id )
        if not aParameter:
            aParameter = aModel.createParameter()
            aParameter.setId( self.id )

        fullID = ecell.ecssupport.createFullID( self.fullID )

        if self.isSetId():
            setSBaseAnnotation( aParameter, 'ID', fullID[ 2 ], aSBMLDocument.getNamespaces() )

        compartment = ecell.ecssupport.createFullIDFromSystemPath( fullID[ 1 ] )
        compartment = ecell.ecssupport.createFullIDString( compartment )
        ( compartment, sbmlType ) \
          = self.theSBMLExporter.searchIdFromFullID( compartment )
        setSBaseAnnotation( aParameter, 'compartment', compartment, aSBMLDocument.getNamespaces() )

        propertyList = self.theSBMLExporter.theEml.getEntityPropertyList( self.fullID )
        for pn in propertyList:
            
            if pn == 'Name':
                name = self.theSBMLExporter.theEml.getEntityProperty( '%s:Name' % ( self.fullID ) )
                if len( name ) != 1:
                    raise SBMLConvertError, \
                          'The format of property [%s:Name] is invalid' \
                          % ( self.fullID )

                name = name[ 0 ]
                aParameter.setName( name )

            elif pn == 'Value':
                value = self.theSBMLExporter.theEml.getEntityProperty( '%s:Value' % ( self.fullID ) )
                if len( value ) != 1:
                    raise SBMLConvertError, \
                          'The format of property [%s:Value] is invalid' \
                          % ( self.fullID )

                value = float( value[ 0 ] )
                aParameter.setValue( value )


            elif pn == 'Fixed':
                value = self.theSBMLExporter.theEml.getEntityProperty( '%s:Fixed' % ( self.fullID ) )
                if len( value ) != 1:
                    raise SBMLConvertError, \
                          'The format of property [%s:Fixed] is invalid' \
                          % ( self.fullID )

                value = int( value[ 0 ] )
                if value == 1:
                    raise SBMLConvertError, 'Property [%s:Fixed] is set as True. It cannot be converted' % ( self.fullID )
                
            else:
                raise SBMLConvertError, \
                      'Property [%s:%s] cannot be converted' \
                      % ( self.fullID, pn )

    # end of writeSBML
    

# end of ParameterExporter


class ReactionExporter( SBaseExporter ):


    def __init__( self, aSBMLExporter, fullIDString, id=None ):

        SBaseExporter.__init__( self, aSBMLExporter, fullIDString, id )

    # end of __init__


    def setFullID( self, fullIDString ):

        fullID = ecell.ecssupport.createFullID( fullIDString )
        if fullID[ 0 ] != ecell.ecssupport.PROCESS:
            raise SBMLConvertError, \
                  'Entity [%s] is not Process' % ( fullIDString )
        
        className = self.theSBMLExporter.theEml.getEntityClass( fullIDString )
        if className != 'ExpressionFluxProcess':
            raise SBMLConvertError, \
                  'Class of [%s] is not ExpressionFluxProcess' \
                  % ( fullIDString )

        propertyList = self.theSBMLExporter.theEml.getEntityPropertyList( fullIDString )
        if not 'Expression' in propertyList:
            raise SBMLConvertError, \
                  'Process [%s] has no Expression' % ( fullIDString )

        SBaseExporter.setFullID( self, fullIDString )

    # end of setFullID
    

    def writeSBML( self, aSBMLDocument ):

        aModel = aSBMLDocument.getModel()
        aReaction = aModel.getReaction( self.id )
        if not aReaction:
            aReaction = aModel.createReaction()
            aReaction.setId( self.id )

        if not aReaction.isSetKineticLaw():
            aKineticLaw = libsbml.KineticLaw()
        else:
            aKineticLaw = aReaction.getKineticLaw()

        fullID = ecell.ecssupport.createFullID( self.fullID )

        if self.isSetId():
            setSBaseAnnotation( aReaction, 'ID', fullID[ 2 ], aSBMLDocument.getNamespaces() )

        compartment = ecell.ecssupport.createFullIDFromSystemPath( fullID[ 1 ] )
        compartment = ecell.ecssupport.createFullIDString( compartment )
        ( compartment, sbmlType ) \
          = self.theSBMLExporter.searchIdFromFullID( compartment )
        setSBaseAnnotation( aReaction, 'compartment', compartment, aSBMLDocument.getNamespaces() )

        anExpression = self.theSBMLExporter.theEml.getEntityProperty( '%s:Expression' % ( self.fullID ) )[ 0 ]
        namespaceDict = self.createNamespaceFromVariableReferenceList()
        anExpressionParser = ExpressionParser( namespaceDict, compartment )
        anASTNode = anExpressionParser.convertExpressionToASTNode( anExpression )

        aKineticLaw.setFormula( libsbml.formulaToString( anASTNode ) )

        parameterList = []
        for i in range( aKineticLaw.getNumParameters() ):
            aParameter = aKineticLaw.getParameter( i )
            if not aParameter.isSetId():
                continue
            parameterList.append( aParameter.getId() )

        reactantList = []
        for i in range( aReaction.getNumReactants() ):
            aReactant = aReaction.getReactant( i )
            if not aReactant.isSetSpecies():
                continue
            reactantList.append( aReactant.getSpecies() )

        productList = []
        for i in range( aReaction.getNumProducts() ):
            aProduct = aReaction.getProduct( i )
            if not aProduct.isSetSpecies():
                continue
            productList.append( aProduct.getSpecies() )

        modifierList = []
        for i in range( aReaction.getNumModifiers() ):
            aModifier = aReaction.getModifier( i )
            if not aModifier.isSetSpecies():
                continue
            modifierList.append( aModifier.getSpecies() )

        for ( id, coeff ) in namespaceDict.values():
            if coeff > 0 and type( id ) == tuple:
                if productList.count( id[ 0 ] ) == 0:
                    aReaction.addProduct( libsbml.SpeciesReference( id[ 0 ], coeff ) )
                else:
                    i = productList.index( id[ 0 ] )
                    aProduct = aReaction.getProduct( i )
                    aProduct.setStoichiometry( coeff )
                    
            elif coeff < 0 and type( id ) == tuple:
                if reactantList.count( id[ 0 ] ) == 0:
                    aReaction.addReactant( libsbml.SpeciesReference( id[ 0 ], -coeff ) )
                else:
                    i = reactantList.index( id[ 0 ] )
                    aReactant = aReaction.getReactant( i )
                    aReactant.setStoichiometry( -coeff )

        for id in anExpressionParser.modifierList:
            if modifierList.count( id ) == 0:
                aReaction.addModifier( libsbml.ModifierSpeciesReference( id ) )

        propertyList = self.theSBMLExporter.theEml.getEntityPropertyList( self.fullID )
        for pn in propertyList:
            
            if pn == 'Name':
                name = self.theSBMLExporter.theEml.getEntityProperty( '%s:Name' % ( self.fullID ) )
                if len( name ) != 1:
                    raise SBMLConvertError, \
                          'The format of property [%s:Name] is invalid' \
                          % ( self.fullID )

                name = name[ 0 ]
                aReaction.setName( name )

            elif pn == 'VariableReferenceList' \
                     or pn == 'Expression' \
                     or pn == 'StepperID':
                pass

            elif pn == 'Priority':
                value = self.theSBMLExporter.theEml.getEntityProperty( '%s:Priority' % ( self.fullID ) )
                if len( value ) != 1:
                    raise SBMLConvertError, \
                          'The type of Parameter [%s:Priority] is invalid' \
                          % ( self.fullID )

                value = int( value[ 0 ] )
                if value != 0:
                    raise SBMLConvertError, 'The Parameter [%s:Priority] is not equal to 0. It cannnot be converted' % ( self.fullID )

            else:
                value = self.theSBMLExporter.theEml.getEntityProperty( '%s:%s' % ( self.fullID, pn ) )
                if len( value ) != 1:
                    raise SBMLConvertError, \
                          'The type of Parameter [%s] is invalid' % ( pn )

                value = float( value[ 0 ] )

                if parameterList.count( pn ) == 0:
                    aParameter = libsbml.Parameter( pn, value )
                    aKineticLaw.addParameter( aParameter )
                else:
                    i = parameterList.index( pn )
                    aParameter = aKineticLaw.getParameter( i )
                    aParameter.setValue( value )

        aReaction.setKineticLaw( aKineticLaw )

    # end of writeSBML


    def createNamespaceFromVariableReferenceList( self ):

        namespaceDict = {}

        propertyList = self.theSBMLExporter.theEml.getEntityPropertyList( self.fullID )
        if not 'VariableReferenceList' in propertyList:
            return namespaceDict

        aVariableReferenceList = self.theSBMLExporter.theEml.getEntityProperty( '%s:VariableReferenceList' % ( self.fullID ) )

        for aVariableReference in aVariableReferenceList:

            fullID = ecell.analysis.util.createVariableReferenceFullID( aVariableReference[ 1 ], self.fullID )
            fullIDString = 'Variable:%s:%s' % ( fullID[ 1 ], fullID[ 2 ] )

            coeff = 0
            if len( aVariableReference ) > 2:
                coeff = int( aVariableReference[ 2 ] )

            ( id, sbmlType ) \
              = self.theSBMLExporter.searchIdFromFullID( fullIDString )

            if sbmlType == libsbml.SBML_SPECIES:
                compartmentId = self.theSBMLExporter.searchIdFromFullID( 'Variable:%s:SIZE' % ( fullID[ 1 ] ) )[ 0 ]
                namespaceDict[ aVariableReference[ 0 ] ] \
                               = ( ( id, compartmentId ), coeff )

            else:
                namespaceDict[ aVariableReference[ 0 ] ] = ( id, coeff )

        return namespaceDict

    # end of createNamespaceFromVariableReferenceList

    
# end of ReactionExporter


class RuleExporter( SBaseExporter ):


    def __init__( self, aSBMLExporter, fullIDString, id=None ):

        SBaseExporter.__init__( self, aSBMLExporter, fullIDString, id )

    # end of __init__


    def initialize( self, aSBMLExporter, fullIDString, id=None ):

        SBaseExporter.initialize( self, aSBMLExporter, fullIDString, id )

        # overwrite the default value
        self.setId( id )

    # end of initialize
    

    def setFullID( self, fullIDString ):

        fullID = ecell.ecssupport.createFullID( fullIDString )
        if fullID[ 0 ] != ecell.ecssupport.PROCESS:
            raise SBMLConvertError, 'Entity [%s] is not Process' % ( fullID )

        className = self.theSBMLExporter.theEml.getEntityClass( fullIDString )
        if className != 'ExpressionAssignmentProcess':
            raise SBMLConvertError, \
                  'Class of [%s] is not ExpressionAssignmentProcess' \
                  % ( fullIDString )

        propertyList = self.theSBMLExporter.theEml.getEntityPropertyList( fullIDString )
        if not 'Expression' in propertyList:
            raise SBMLConvertError, \
                  'Process [%s] has no Expression' % ( fullIDString )

        SBaseExporter.setFullID( self, fullIDString )

    # end of setFullID
    

    def writeSBML( self, aSBMLDocument ):

        aModel = aSBMLDocument.getModel()
        if self.id == None:
            aRule = aModel.createAssignmentRule()
        else:
            aRule = aModel.getRule( self.id )
            if not aRule:
                raise SBMLConvertError, 'Rule [%s] is not found' % ( self.id )

        namespaceDict = self.createNamespaceFromVariableReferenceList()

        variableName = ''
        assignmentList = []
        for referenceName in namespaceDict.keys():
            ( sbmlId, coeff ) = namespaceDict[ referenceName ]
            if coeff != 0:
                assignmentList.append( referenceName )

        if len( assignmentList ) != 1:
            raise SBMLConvertError, 'Variable for AssignmentRule(ExpressionAssignmentProcess) [%s] is not unique.' % ( self.fullID )
    
        ( id, coeff ) = namespaceDict[ assignmentList[ 0 ] ]
        if coeff != 1:
            raise SBMLConvertError, 'Coefficient for AssignmentRule(ExpressionAssignmentProcess) [%s] must be equal to 1.' % ( self.fullID )

        if type( id ) == tuple:
            variableName = id[ 0 ]
        else:
            variableName = id

        aRule.setVariable( variableName )

        fullID = ecell.ecssupport.createFullID( self.fullID )

        compartment = ecell.ecssupport.createFullIDFromSystemPath( fullID[ 1 ] )
        compartment = ecell.ecssupport.createFullIDString( compartment )
        ( compartment, sbmlType ) \
          = self.theSBMLExporter.searchIdFromFullID( compartment )

        aXMLNamespaceList = aSBMLDocument.getNamespaces()
        setSBaseAnnotation( aRule, 'ID', fullID[ 2 ], aXMLNamespaceList )
        setSBaseAnnotation( aRule, 'compartment', compartment, aXMLNamespaceList )

        anExpression = self.theSBMLExporter.theEml.getEntityProperty( '%s:Expression' % ( self.fullID ) )[ 0 ]
        anExpressionParser = ExpressionParser( namespaceDict )
        anASTNode = anExpressionParser.convertExpressionToASTNode( anExpression )

        aRule.setFormula( libsbml.formulaToString( anASTNode ) )

        propertyList = self.theSBMLExporter.theEml.getEntityPropertyList( self.fullID )
        for pn in propertyList:
            
            if pn == 'Name':
                name = self.theSBMLExporter.theEml.getEntityProperty( '%s:Name' % ( self.fullID ) )
                if len( name ) != 1:
                    raise SBMLConvertError, \
                          'The format of property [%s:Name] is invalid' \
                          % ( self.fullID )

                name = name[ 0 ]
                setSBaseAnnotation( aRule, 'Name', name, aXMLNamespaceList )

            elif pn == 'Priority':
                value = self.theSBMLExporter.theEml.getEntityProperty( '%s:Priority' % ( self.fullID ) )
                if len( value ) != 1:
                    raise SBMLConvertError, \
                          'The type of Parameter [%s:Priority] is invalid' \
                          % ( self.fullID )

                value = int( value[ 0 ] )
                if value != 0:
                    raise SBMLConvertError, 'The Parameter [%s:Priority] is not equal to 0. It cannnot be converted' % ( self.fullID )

            elif pn == 'VariableReferenceList' \
                     or pn == 'Expression' \
                     or pn == 'StepperID':
                pass

            else:
                raise SBMLConvertError, 'Property [%s] is not supported for ExpressionAssignmentProcess. Define it as a global parameter' % ( pn )

    # end of writeSBML

    
    def createNamespaceFromVariableReferenceList( self ):

        namespaceDict = {}

        propertyList = self.theSBMLExporter.theEml.getEntityPropertyList( self.fullID )
        if not 'VariableReferenceList' in propertyList:
            return namespaceDict

        aVariableReferenceList = self.theSBMLExporter.theEml.getEntityProperty( '%s:VariableReferenceList' % ( self.fullID ) )

        for aVariableReference in aVariableReferenceList:

            fullID = ecell.analysis.util.createVariableReferenceFullID( aVariableReference[ 1 ], self.fullID )
            fullIDString = 'Variable:%s:%s' % ( fullID[ 1 ], fullID[ 2 ] )

            coeff = 0
            if len( aVariableReference ) > 2:
                coeff = int( aVariableReference[ 2 ] )

            ( id, sbmlType ) \
              = self.theSBMLExporter.searchIdFromFullID( fullIDString )

            if sbmlType == libsbml.SBML_SPECIES:
                compartmentId = self.theSBMLExporter.searchIdFromFullID( 'Variable:%s:SIZE' % ( fullID[ 1 ] ) )[ 0 ]
                namespaceDict[ aVariableReference[ 0 ] ] \
                               = ( ( id, compartmentId ), coeff )

            else:
                namespaceDict[ aVariableReference[ 0 ] ] = ( id, coeff )

        return namespaceDict

    # end of createNamespaceFromVariableReferenceList

    
# end of RuleExporter


if __name__ == '__main__':

    import sys
    import os.path
    
    import SBMLExporter


    def main( emlfilename, sbmlfilename=None ):
        
        ( basenameString, extString ) \
          = os.path.splitext( os.path.basename( emlfilename ) )

        aSBMLExporter = SBMLExporter.SBMLExporter( emlfilename )
        if sbmlfilename != None:
            aSBMLExporter.loadSBML( sbmlfilename )

        i = 1
        while os.path.isfile( '%s%d.xml' % ( basenameString, i ) ):
            i += 1
        aSBMLExporter.saveAsSBML( '%s%d.xml' % ( basenameString, i ) )

##         aSBMLDocument = aSBMLExporter.convertEMLToSBML()
##         print aSBMLDocument

    # end of main
    

    if len( sys.argv ) == 2:
        main( sys.argv[ 1 ] )
    elif len( sys.argv ) == 3:
        main( sys.argv[ 1 ], sys.argv[ 2 ] )
