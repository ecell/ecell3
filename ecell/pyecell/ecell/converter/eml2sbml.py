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

# Eml to Sbml converter

import sys
import string
import os
import types
import getopt
import time
from sets import Set
import libsbml
import numpy

from ecell.ecssupport import *
from ecell.ecs_constants import *
from ecell.eml import *
import ecell.expr.lexer as exprlexer
import ecell.expr.parser as exprparser
from ecell.converter.sbml.utils import *
import ecell.util as util

__all__ = (
    'convert'
    )

class IDManager:
    def __init__( self ):
        self.theMap = {}
        self.theSBMLIDNamespace = Set()

    def get( self, aFullID ):
        assert aFullID[1] != '.' # relative pathes are not allowed
        aKey = aFullID[1] + ':' + aFullID[2]
        if self.theMap.has_key( aKey ):
            return self.theMap[ aKey ]
        if aFullID[1] == '' and ( aFullID[2] == '' or aFullID[2] == '/' ):
            return 'default'
        else:
            return aFullID[2]

    def put( self, aFullID ):
        assert aFullID[1] != '.' # relative pathes are not allowed
        aKey = aFullID[1] + ':' + aFullID[2]
        if self.theMap.has_key( aKey ):
            return self.theMap[ aKey ]
        if aFullID[2] in self.theSBMLIDNamespace:
            if aFullID[1] == '/' or aFullID[1] == '':
                aSBMLID = 'default__' + aFullID[2]
            else:
                aSBMLID = aFullID[1].rstrip('/').replace('/', '__') \
                          + '__' + aFullID[2]
        else:
            if aFullID[1] == '' and ( aFullID[2] == '' or aFullID[2] == '/'):
                aSBMLID = 'default'
            else:
                aSBMLID = aFullID[2]
        self.theMap[ aKey ] = aSBMLID
        self.theSBMLIDNamespace.add( aSBMLID )
        return aSBMLID

    def hasSBMLID( self, aSBMLID ):
        return aSBMLID in self.theSBMLIDNamespace

class SBMLExprRendererError( Exception ):
    pass

class SBMLExpression:
    def __init__( self, aExprString ):
        self.theExprString = aExprString

    def __str__( self ):
        return self.theExprString

class EntityRef:
    def getPropery( self, aName ):
        raise SBMLExprRendererError, "No such attribute in %s: `%s'" % \
            ( self.__class__, aName )

    def invoke( self, aName, aParamList ):
        raise SBMLExprRendererError, "No such method in %s: `%s'" % \
            ( self.__class__, aName )

class SystemRef( EntityRef ):
    def __init__( self, aRenderer, aSystemPath ):
        self.theRenderer = aRenderer
        self.theSystemPath = aSystemPath

    def getSystemPath( self ):
        return self.theSystemPath

    def getCompartmentID( self ):
        return self.theRenderer.theIDManager.get(
            util.convertSystemPathToFullID( self.theSystemPath ) )

    def getProperty( self, aName ):
        if aName == 'Size':
            return SBMLExpression( self.getCompartmentID() )
        elif aName == 'SizeN_A':
            return SBMLExpression( '(' + self.getCompartmentID() + '*N_A)' )
        else:
            raise SBMLExprRendererError, "getSuperSystem attribute must be Size or SizeN_A"

    def invoke( self, aName, aParamList ): 
        if aName == 'getSuperSystem':
            if len( aParamList ) != 0:
                raise ValueError, "Wrong number of arguments"
            aSuperSystemPath = util.getSuperSystemPath( self.theSystemPath )
            if aSuperSystemPath == None:
                return None
            return self.theRenderer.getSystemRef( aSuperSystemPath )
        else:
            return EntityRef.invoke( self, aName, aParamList )

class ProcessRef( EntityRef ):
    def __init__( self, aRenderer, aSystemRef, params ):
        self.theRenderer = aRenderer
        self.theSystemRef = aSystemRef
        self.params = params

    def getProperty( self, aName ):
        if not self.params.has_key( aName ):
            return EntityRef.getProperty( self, aName )
        return self.params[ aName ]

    def invoke( self, aName, aParamList ): 
        if aName == 'getSuperSystem':
            if len( aParamList ) != 0:
                raise ValueError, "Wrong number of arguments"
            return self.theSystemRef
        else:
            return EntityRef.invoke( self, aName, aParamList )

class VarRef( EntityRef ):
    def __init__( self, aRenderer, aSystemRef, anID ):
        self.theRenderer = aRenderer
        self.theSystemRef = aSystemRef
        self.theID = anID

    def getProperty( self, aName ):
        # in case of Compartment
        aPath = self.theSystemRef.getSystemPath()
        if self.theID == 'SIZE': 
            return self.theSystemRef.getProperty('Size')
        # in case of Species
        aVariableID = self.theRenderer.theIDManager.put(
            util.createFullID_M( VARIABLE, aPath, self.theID ) )

        if aName == 'Value':
            return SBMLExpression(
                '(' + aVariableID + '/' + aSystem.getCompartmentID() + '/N_A)' )
        elif aName == 'NumberConc':
            return SBMLExpression( '(' + aVariableID + '/N_A)' )
        elif aName == 'MolarConc':
            return SBMLExpression( aVariableID )
        else:
            raise SBMLExprRendererError, \
                "VariableReference attribute must be one of MolarConc, NumberConc and Value"

class SBMLExprRenderer:
    def __init__( self, anIDManager, aReactionPath, varRefs, params ):
        self.theIDManager = anIDManager
        self.theReactionPath = aReactionPath
        self.varRefs = varRefs
        self.params = params
        self.sysRefs = {}

    def getSystemRef( self, aSystemPath ):
        if self.sysRefs.has_key( aSystemPath ):
            return self.sysRefs[ aSystemPath ]
        else:
            aSysRef = SystemRef( self, aSystemPath )
            self.sysRefs[ aSystemPath ] = aSysRef
            return aSysRef

    def resolveVarRef( self, aVariableName ):
        if aVariableName == 'self':
            return ProcessRef( self,
                self.getSystemRef( self.theReactionPath ),
                self.params )

        if self.params.has_key( aVariableName ):
            return SBMLExpression( aVariableName )

        if not self.varRefs.has_key( aVariableName ):
            raise NameError, "Could not find variable reference `%s'" % aVariableName
        aType, aSystemPath, aVariable = self.varRefs[ aVariableName ]
        # If there are some var_ref which call getValue()
        # function "ID.Value", this var_ref need to be 
        # distinguished between [Species] and [Parameter].
        #
        # If it is the [Species], it must be converted into
        # MolarConcentration.  "ID.Value / ( SIZE * N_A )"
        #
        # In case of [Parameter], it must be without change.
        # var_ref attribute is Value
        anID = 'SBMLParameter__' + aVariable
        if aSystemPath == '/SBMLParameter' or \
           ( aSystemPath == '.' and \
             self.theReactionPath == '/SBMLParameter' ):
            if self.theIDManager.hasSBMLID( anID ):
                aVariable = anID
        else:
            if aSystemPath == '.':
                if self.theReactionPath == '/':
                    aSystemRef = self.getSystemRef( '/' )
                else:
                    aSystemRef = self.getSystemRef(
                        self.theReactionPath )
            else:
                if aSystemPath == '/':
                    aSystemRef = self.getSystemRef( '/' )
                else:
                    aSystemRef = self.getSystemRef(
                        aSystemPath )
        return VarRef( self, aSystemRef, aVariable )

    def getPrecedence( aNode ):
        if aNode == None:
            return 0
        elif isinstance( aNode, exprparser.AddOpNode ) or \
           isinstance( aNode, exprparser.SubOpNode ):
            return 1
        elif isinstance( aNode, exprparser.MulOpNode ) or \
             isinstance( aNode, exprparser.DivOpNode ):
            return 2
        elif isinstance( aNode, exprparser.DerefOpNode ):
            return 3
        elif isinstance( aNode, exprparser.NegOpNode ):
            return 4
        else:
            return 5
    getPrecedence = staticmethod( getPrecedence )

    def render( self, aNode ):
        return str( self.eval( self._render( aNode ) ) )

    def eval( self, exprs ):
        if isinstance( exprs, SBMLExpression ):
            return exprs
        elif type( exprs ) == list:
            image = ''
            for expr in exprs:
                image += str( self.eval( expr ) )
            return SBMLExpression( image )
        else:
            raise SBMLExprRendererError, "Unevaluable expression"

    def _render( self, aNode, aParentNode = None ):
        retval = None
        if isinstance( aNode, exprparser.AddOpNode ):
            retval = [
                self._render( aNode[0], aNode ),
                SBMLExpression( '+' ),
                self._render( aNode[1], aNode )
                ]
        elif isinstance( aNode, exprparser.SubOpNode ):
            retval = [
                self._render( aNode[0], aNode ),
                SBMLExpression( '-' ),
                self._render( aNode[1], aNode )
                ]
        elif isinstance( aNode, exprparser.MulOpNode ):
            retval = [
                self._render( aNode[0], aNode ),
                SBMLExpression( '*' ),
                self._render( aNode[1], aNode )
                ]
        elif isinstance( aNode, exprparser.DivOpNode ):
            retval = [
                self._render( aNode[0], aNode ),
                SBMLExpression( '/' ),
                self._render( aNode[1], aNode )
                ]
        elif isinstance( aNode, exprparser.NegOpNode ):
            retval = [
                SBMLExpression( '-' ),
                self._render( aNode[0], aNode )
                ]
        elif isinstance( aNode, exprparser.ScalarNode ):
            retval = aNode[0]
        elif isinstance( aNode, exprparser.VarRefNode ):
            retval = self.resolveVarRef( aNode[0] )
        elif isinstance( aNode, exprparser.IdentifierNode ):
            retval = SBMLExpression( aNode[0] )
        elif isinstance( aNode, exprparser.FunctionCallNode ):
            aName = self._render( aNode[0] )
            aParamList = map( self.eval, map( self._render, aNode[1] ) )
            retval = [
                aName,
                SBMLExpression( '(' ),
                ] + aParamList +  [
                SBMLExpression( ')' )
                ]
        elif isinstance( aNode, exprparser.DerefOpNode ):
            aTarget = self._render( aNode[0], aNode )
            aName = self._render( aNode[1], aNode )
            if not isinstance( aName, SBMLExpression ):
                raise SBMLExprRendererError, "Dynamic dereference not allowed"
            if isinstance( aTarget, EntityRef ):
                retval = aTarget.getProperty( str( aName ) )
            else:
                retval = [
                    aTarget,
                    SBMLExpression( '.' ),
                    aName
                    ]
        elif isinstance( aNode, exprparser.MethodNode ):
            aTarget = self._render( aNode[0], aNode )
            aName = self._render( aNode[1][0] )
            aParamList = map( self.eval, map( self._render, aNode[1][1] ) )
            if not isinstance( aName, SBMLExpression ):
                raise SBMLExprRendererError, "Dynamic invocation not allowed"
            if isinstance( aTarget, EntityRef ):
                retval = aTarget.invoke( str( aName ), aParamList )
            else:
                retval = [
                    aTarget,
                    SBMLExpression( '.' ),
                    aName
                    ]
        if self.getPrecedence( aNode ) < self.getPrecedence( aParentNode ):
            retval = [
                SBMLExpression( '(' ),
                retval,
                SBMLExpression( ')' )
                ]
        return retval

class SBMLRenderer:
    def __init__( self, aSBMLDocument ):
        self.theSBMLModel = aSBMLDocument.getModel()
        self.theSBMLLevel = aSBMLDocument.getLevel()
        self.theIDManager = IDManager()

    def markDelayNode( anASTNode ):
        """
        if Delay function "delay(,)" is defined in Expression,
        AST Node must be marked for recognizing it is "csymbol"
        """
        aNumChildren = anASTNode.getNumChildren()
        
        if aNumChildren == 2:
            if anASTNode.isFunction() == True and \
                 anASTNode.getName() == 'delay':
                anASTNode.setType( libsbml.AST_FUNCTION_DELAY )
            SBMLRenderer.markDelayNode( anASTNode.getLeftChild() )
            SBMLRenderer.markDelayNode( anASTNode.getRightChild() )
        elif aNumChildren == 1:
            SBMLRenderer.markDelayNode( anASTNode.getLeftChild() )
    markDelayNode = staticmethod( markDelayNode )

    def getVariableReferenceId( self, aVariableReference, aCurrentSystem ):
        """
        return the VariableReference ID
        """
        aFullID = util.createFullID( aVariableReference )
        aSystemPath = util.getSystemPath( aFullID )
        if aSystemPath == '.':
            aSystemPath = aCurrentSystem
            aFullID = util.createFullID_M( aFullID[0], aCurrentSystem, aFullID[2]  )
        aSystemFullID = util.convertSystemPathToFullID( aSystemPath )
        if aFullID[ID] == 'SIZE' and aSystemPath != '/SBMLParameter':
            return self.theIDManager.put( aSystemFullID )
        else:
            return self.theIDManager.put( aFullID )

    def convertExpression( self, anExpression, aVariableReferenceList, aReactionPath, aProcessAttributeMap ):
        aDelayFlag = False

        aParser = exprparser.createParser()
        anASTNode = aParser.parse( anExpression,
            lexer = exprlexer.createLexer() )

        aRenderer = SBMLExprRenderer( self.theIDManager,
            aReactionPath,
            dict( map( lambda x: (x[0], createFullID( x[1] ) ),
                aVariableReferenceList ) ),
            aProcessAttributeMap )
        return aRenderer.render( anASTNode )

    def createEntity( self, anEml, aFullID, anOptional='' ):
        """
        create Compartment, Species, Parameter and Reaction object
        """
        aFullIDString = util.createFullIDString( aFullID )
        aType = ENTITYTYPE_STRING_LIST[ aFullID[ TYPE ] ]

        aClass = anEml.getEntityClass( aFullIDString )

        # -- make the Species and the Parameter
        if aFullID[TYPE] == VARIABLE:
            aPropertyMap = anEml.getEntityPropertyMap( aFullIDString )
            aCurrentCompartment = self.theIDManager.put(
                util.convertSystemPathToFullID(
                    util.getSystemPath( aFullID ) ) )
            aCurrentCompartmentObj = self.theSBMLModel.getCompartment(
                aCurrentCompartment )
            if aCurrentCompartment == "SBMLParameter":
                # create Parameter object
                aParameter = self.theSBMLModel.createParameter()
                aPropertyName = util.getPropertyName( aFullID )
                aParameterID = self.theIDManager.put(
                    util.createFullID_M(
                        VARIABLE, 'SBMLParameter', aPropertyName ) )
                # set Parameter ID
                if self.theSBMLLevel == 1:
                    aParameter.setName( aParameterID )
                elif self.theSBMLLevel == 2:
                    aParameter.setId( aParameterID )
                # set Parameter Name, Value and Constant
                for aPropertyName in aPropertyMap:
                    # set Parameter Name
                    if aPropertyName == "Name":
                        if self.theSBMLLevel == 1:
                            pass
                        if self.theSBMLLevel == 2:
                            aParameter.setName( aPropertyMap[ aPropertyName ] )
                    # set Parameter Value
                    elif aPropertyName == "Value":
                        aParameter.setValue(
                            float( aPropertyMap[ aPropertyName ] ) )
                    # set Constant 
                    elif ( aPropertyName == "Fixed" ):
                        aParameter.setConstant(
                            int( float( aPropertyMap[ aPropertyName ] ) ) )
                    else:
                        raise AttributeError, \
                            "Unrepresentable property `%s' in Parameter" % \
                            aPropertyName
            else:
                if aFullID[2] != "SIZE" and aFullID[2] != "Dimensions":
                    # create Species object
                    aSpecies = self.theSBMLModel.createSpecies()
                    # set Species ID
                    aSpeciesID = self.theIDManager.put( aFullID )
                    if self.theSBMLLevel == 1:
                        aSpecies.setName( aSpeciesID )
                    if self.theSBMLLevel == 2:
                        aSpecies.setId( aSpeciesID )
                    # set Compartment of Species
                    if aCurrentCompartment == '':
                        aSpecies.setCompartment( 'default' )
                    else:
                        aSpecies.setCompartment( aCurrentCompartment )
                    # set Species Name, Value and Constant
                    for aPropertyName in aPropertyMap:
                        # set Species Name
                        if aPropertyName == "Name":
                            if self.theSBMLLevel == 1:
                                pass
                            elif self.theSBMLLevel == 2:
                                aSpecies.setName(
                                    aPropertyMap[ aPropertyName ] )
                        # set Species Value
                        elif aPropertyName == "Value":
                            aSpecies.setInitialAmount( float(
                                aPropertyMap[ aPropertyName ] ) / N_A )
                        # set Species Constant
                        elif ( aPropertyName == "Fixed" ):
                            aSpecies.setConstant(
                                int( aPropertyMap[ aPropertyName ] ) )
                        # set Concentration by rule
                        elif ( aPropertyName == "MolarConc" ):
                            # XXX: units are just eventually correct here, because
                            # SBML falls back to mole and liter for substance and
                            # volume of the species if these are unspecified.
                            if self.theSBMLLevel == 1:
                                if aCurrentCompartmentObj != None:
                                    compVol = float(
                                        aCurrentCompartmentObj.getVolume() )
                                else:
                                    compVol = 1.0
                                propValue = float(
                                    aPropertyMap[ aPropertyName ] )
                                aSpecies.setInitialAmount( compVol * propValue )
                            else: # SBML lv.2
                                aSpecies.setInitialConcentration(
                                    float( aPropertyMap[ aPropertyName ] )
                                    )
                        else:
                            raise AttributeError, \
                                "Unrepresentable property `%s' in Species" % \
                                    aPropertyName
        # -- make the Reaction and the Rule object
        elif aFullID[TYPE] == PROCESS:
            aPropertyMap = anEml.getEntityPropertyMap( aFullIDString )
            aType, aSystemPath, aProcessID = createFullID( aFullIDString )

            # -- make Rule object
            if util.getSystemPath( aFullID ) == '/SBMLRule':
                # get Process Class
                aProcessClass = anEml.getEntityClass( aFullIDString )
                aVariableReferenceList = aPropertyMap[ 'VariableReferenceList' ]
                anExpression = self.convertExpression(
                    aPropertyMap[ 'Expression' ],
                    aVariableReferenceList,
                    util.getSystemPath( aFullID ),
                    aPropertyMap
                    )
                if aProcessClass == 'ExpressionAlgebraicProcess':
                    # create AlgebraicRule object
                    anAlgebraicRule = self.theSBMLModel.createAlgebraicRule()
                    # set AlgebraicRule Formula
                    anAlgebraicRule.setFormula( anExpression )
                elif aProcessClass == 'ExpressionAssignmentProcess':
                    for aVariableReference in aVariableReferenceList:
                        if len( aVariableReference ) >= 3:
                            if aVariableReference[2] != '0': 
                                # create AssignmentRule object
                                anAssignmentRule =self.theSBMLModel.createAssignmentRule()
                                # set AssignmentRule Formula
                                anAssignmentRule.setFormula(
                                    aVariableReference[2] + '* ( ' + anExpression + ')' )
                                aVariableID = self.getVariableReferenceId(
                                    aVariableReference[1], aFullID[1] )
                                anAssignmentRule.setVariable( aVariableID )
                elif aProcessClass == 'ExpressionFluxProcess':
                    for aVariableReference in aVariableReferenceList:
                        if len( aVariableReference ) >= 3:
                            if aVariableReference[2] != '0': 
                                # create AssignmentRule object
                                aRateRule = self.theSBMLModel.createRateRule()
                                # set AssignmentRule Formula
                                aRateRule.setFormula(
                                    aVariableReference[2] + \
                                    '* ( ' + anExpression + ')' )
                                aVariableID = self.getVariableReferenceId(
                                    aVariableReference[1], aFullID[1] )
                                aRateRule.setVariable( aVariableID )
                else:
                    raise TypeError, \
                        "The type of Process must be Algebraic, Assignment, Flux Processes"
            # -- make Reaction object
            else:
                # create Parameter object
                aReaction = self.theSBMLModel.createReaction()
                # create KineticLaw Object
                aKineticLaw = self.theSBMLModel.createKineticLaw()
                # set Reaction ID
                if self.theSBMLLevel == 1:
                    aReaction.setName( aFullID[2] )
                if self.theSBMLLevel == 2:
                    aReaction.setId( aFullID[2] )
                for aPropertyName in aPropertyMap:
                    # set Name property ( Name )
                    if aPropertyName == "Name":
                        # set Reaction Name
                        if self.theSBMLLevel == 1:
                            pass
                        if self.theSBMLLevel == 2:
                            aReaction.setName( aPropertyMap[ aPropertyName ] )
                    # set Expression property ( KineticLaw Formula )
                    elif aPropertyName == "Expression":
                        # convert Expression of the ECELL format to
                        # SBML kineticLaw formula
                        anExpression = aPropertyMap[ aPropertyName ]
                        aVariableReferenceList = \
                            aPropertyMap[ 'VariableReferenceList' ]
                        anExpression = self.convertExpression(
                            anExpression,
                            aVariableReferenceList,
                            util.getSystemPath( aFullID ),
                            aPropertyMap
                            )
                        # get Current System Id
                        for aVariableReference in aVariableReferenceList:
                            if len( aVariableReference ) == 3:
                                if int( aVariableReference[2] ) != 0:
                                    aFirstColon =\
                                        string.index( aVariableReference[1], ':' )
                                    aLastColon =\
                                        string.rindex( aVariableReference[1], ':' )
                                    if aVariableReference[1][aFirstColon+1:aLastColon] == '.':
                                        aLastSlash =\
                                            string.rindex( aFullID[1], '/' )
                                        CompartmentOfReaction=\
                                            aFullID[1][aLastSlash+1:]
                                    else: 
                                        aLastSlash =\
                                            string.rindex( aVariableReference[1], '/' )
                                        CompartmentOfReaction=\
                                            aVariableReference[1][aLastSlash+1:aLastColon]
                        if CompartmentOfReaction == '':
                            anExpression = '(' + anExpression + ')/default/N_A'
                        else:
                            anExpression =\
                                '(' + anExpression + ')/' + \
                                CompartmentOfReaction + '/N_A'
                        # set KineticLaw Formula
                        anASTNode = libsbml.parseFormula( anExpression )
                        self.markDelayNode( anASTNode )
                        aKineticLaw.setMath( anASTNode )
                    # set VariableReference property ( SpeciesReference )
                    elif aPropertyName == "VariableReferenceList":
                        # make a flag. Because SBML model is defined
                        # both Product and Reactant. This flag is required
                        # in order to judge whether the Product and the
                        # Reactant are defined.

                        aReactantFlag = False
                        aProductFlag = False
                        for aVariableReference in aPropertyMap[ aPropertyName ]:
                            if len( aVariableReference ) >= 3:
                                # add Reactants to Reaction object
                                if float( aVariableReference[2] ) < 0:
                                    # change the Reactant Flag
                                    aReactantFlag = True
                                    # create Reactant object
                                    aReactant = self.theSBMLModel.createReactant()
                                    # set Species Id to Reactant object
                                    aSpeciesReferenceId = self.getVariableReferenceId(
                                        aVariableReference[1], aFullID[1] )
                                    # XXX: Unicode to native conversion
                                    aReactant.setSpecies( str( aSpeciesReferenceId ) )
                                    # set Stoichiometry 
                                    aReactant.setStoichiometry(
                                        -( float( aVariableReference[2] ) ) )
                                # add Products to Reaction object
                                elif float( aVariableReference[2] ) > 0:
                                    # change the Product Flag
                                    aProductFlag = True
                                    # create Product object
                                    aProduct = self.theSBMLModel.createProduct()
                                    # set Species Id
                                    aSpeciesReferenceId = self.getVariableReferenceId(
                                        aVariableReference[1], aFullID[1] )
                                    # XXX: Unicode to native conversion
                                    aProduct.setSpecies( str ( aSpeciesReferenceId ) )
                                    # set Stoichiometry
                                    aProduct.setStoichiometry(
                                        float( aVariableReference[2] ) )
                                # add Modifiers to Reaction object
                                else:
                                    # create Modifier object
                                    aModifier = self.theSBMLModel.createModifier()
                                    
                                    # set Species Id to Modifier object
                                    aVariableReferenceId = self.getVariableReferenceId(
                                        aVariableReference[1], aFullID[1] )
                                    # XXX: Unicode to native conversion
                                    aModifier.setSpecies( str( aVariableReferenceId ) )
                            # if there isn't the stoichiometry
                            elif len( aVariableReference ) == 2:
                                # create Modifier object
                                aModifier = self.theSBMLModel.createModifier()
                                # set Species Id to Modifier object
                                aVariableReferenceId = self.getVariableReferenceId(
                                    aVariableReference[1], aFullID[1] )
                                # XXX: Unicode to native conversion
                                aModifier.setSpecies( str( aVariableReferenceId ) )
                        if aReactantFlag == False or aProductFlag == False:
                            # set EmptySet Species, because if it didn't define,
                            # Reactant or Product can not be defined.
                            if aReactantFlag == False:
                                # create Reactant object
                                aReactant = self.theSBMLModel.createReactant()
                                # set Species Id to Reactant object
                                aReactant.setSpecies( 'EmptySet' )
                                # set Stoichiometry 
                                aReactant.setStoichiometry( 0 )
                            elif aProductFlag == False:
                                # create Product object
                                aProduct = self.theSBMLModel.createProduct()
                                # set Species Id
                                aProduct.setSpecies( 'EmptySet' )
                                # set Stoichiometry
                                aProduct.setStoichiometry( 0 )
                    # These properties are not defined in SBML Lv2
                    elif aPropertyName == "Priority" or \
                         aPropertyName == "Activity" or \
                         aPropertyName == "IsContinuous" or \
                         aPropertyName == "StepperID" or \
                         aPropertyName == "FireMethod" or \
                         aPropertyName == "InitializeMethod":
                        pass
                    else:
                        # create Parameter Object (Local)
                        aParameter = self.theSBMLModel.createKineticLawParameter()
                        # set Parameter ID
                        # XXX: Unicode => native conversion
                        aParameter.setId( str( aPropertyName ) )
                        # set Parameter Value
                        aParameter.setValue(
                            float( aPropertyMap[ aPropertyName ] ) )
                # add KineticLaw Object to Reaction Object
                aReaction.setKineticLaw( aKineticLaw )
        # -- make the Compartment 
        elif aFullID[TYPE] == SYSTEM:
            if aFullID[2] != 'SBMLParameter' and aFullID[2] != 'SBMLRule':
                # create Compartment object
                aCompartment = self.theSBMLModel.createCompartment()
                aCompartmentID = self.theIDManager.put( aFullID )
                if self.theSBMLLevel == 1:
                    # XXX: Unicode to native conversion
                    aCompartment.setName( str( aCompartmentID ) )
                elif self.theSBMLLevel == 2:
                    # XXX: Unicode to native conversion
                    aCompartment.setId( str( aCompartmentID ) )

                aSystemPath = convertFullIDToSystemPath( aFullID )

                for anID in anEml.getEntityList( 'Variable', aSystemPath ):
                    # set Size and constant of Compartment
                    if anID == "SIZE":
                        tmpPropertyMap = anEml.getEntityPropertyMap(
                            "Variable:" + aSystemPath + ":SIZE" )
                        if tmpPropertyMap.has_key( "Value" ):
                            aCompartment.setSize( float(
                                tmpPropertyMap[ "Value"] ) )
                        if tmpPropertyMap.has_key( "Fixed" ):
                            aCompartment.setConstant(
                                int( tmpPropertyMap[ "Fixed" ] ) )
                    # set Dimensions of Compartment
                    elif anID == "Dimensions":
                        aFullPN = "Variable:" + aSystemPath + ':' + anID + ":Value"
                        aCompartment.setSpatialDimensions(
                            int( anEml.getEntityProperty( aFullPN ) ) )
                # set Outside element of Compartment
                anOutsideComponentID = self.theIDManager.put(
                    util.convertSystemPathToFullID(
                        util.getSystemPath( aFullID ) ) )
                aCompartment.setOutside( anOutsideComponentID )

    def renderSystemContent( self, anEml, aSystemPath ):
        aFullID = util.convertSystemPathToFullID( aSystemPath )
        self.createEntity( anEml, aFullID )
        # set Species
        for anID in anEml.getEntityList( 'Variable', aSystemPath ):
            aFullID = util.createFullID_M( VARIABLE, aSystemPath, anID )
            self.createEntity( anEml, aFullID )
        # set Reaction
        for anID in anEml.getEntityList( 'Process', aSystemPath ):
            aFullID = util.createFullID_M( PROCESS, aSystemPath, anID )
            self.createEntity( anEml, aFullID )
        # create SubSystem by iterating calling createModel
        for aSystem in anEml.getEntityList( 'System', aSystemPath ):
            aSubSystemPath = joinSystemPath( aSystemPath, aSystem )
            self.renderSystemContent( anEml, aSubSystemPath )

    def render( self, anEml ):
        self.renderSystemContent( anEml, '/' )
        # set N_A Parameter
        isAbogadroNumber = False
        if self.theSBMLLevel == 1:
            for aParameter in getParameter( self.theSBMLModel ):
                if( aParameter[1] == 'N_A' ):
                    isAbogadroNumber = True
            if not( isAbogadroNumber ):
                # create Parameter object
                aParameter = self.theSBMLModel.createParameter()
                # set Parameter Name
                aParameter.setName( 'N_A' )
                # set Parameter Value
                aParameter.setValue( float( N_A ) )
                # set Parameter Constant
                aParameter.setConstant( int( 1 ) )
        elif self.theSBMLLevel == 2:
            hasAbogadroNumber = False
            for aParameter in getParameter( self.theSBMLModel ):
                if aParameter[0] == 'N_A':
                    hasAbogadroNumber = True
                    break
            if not hasAbogadroNumber:
                # create Parameter object
                aParameter = self.theSBMLModel.createParameter()
                # set Parameter ID
                aParameter.setId( 'N_A' )
                # set Parameter Value
                aParameter.setValue( float( N_A ) )
                # set Parameter Constant
                aParameter.setConstant( int( 1 ) )
        # set EmptySet
        isEmptySet = False
        if self.theSBMLLevel == 1:
            for aSpecies in getSpecies( self.theSBMLModel ):
                if aSpecies[1] == 'EmptySet':
                    isEmptySet = True
            if not isEmptySet:
                # create Species object
                aSpecies = self.theSBMLModel.createSpecies()
                # set Species Name
                aSpecies.setName( 'EmptySet' )
                # set Species Compartment
                aSpecies.setCompartment( 'default' )
                # set Species Amount
                aSpecies.setInitialAmount( float( 0 ) )
                # set Species Constant
                aSpecies.setConstant( int( 1 ) )    
        elif self.theSBMLLevel == 2:
            for aSpecies in getSpecies( self.theSBMLModel ):
                if aSpecies[0] == 'EmptySet':
                    isEmptySet = True
            if not isEmptySet:
                # create Species object
                aSpecies = self.theSBMLModel.createSpecies()
                # set Species Id
                aSpecies.setId( 'EmptySet' )
                # set Species Compartment
                aSpecies.setCompartment( 'default' )
                # set Species Amount
                aSpecies.setInitialAmount( float( 0 ) )
                # set Species Constant
                aSpecies.setConstant( int( 1 ) )    

def convert( anEmlContent, basename, level, version ):
    level = int( level )
    version = int( version )

    anSBMLDoc = libsbml.SBMLDocument( level, version )
    anSBMLDoc.createModel().setId( basename )

    SBMLRenderer( anSBMLDoc ).render( Eml( anEmlContent ) )
    return libsbml.writeSBMLToString( anSBMLDoc )
