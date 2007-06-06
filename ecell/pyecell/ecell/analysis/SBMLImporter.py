#!/usr/bin/env python
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
"""

__program__ = 'SBMLImporter'
__version__ = '1.0'
__author__ = 'Kazunari Kaizu <kaizu@sfc.keio.ac.jp>'
__copyright__ = ''
__license__ = ''


import string
import copy

import libsbml

import ecell.eml
import ecell.ecssupport
import ecell.ecs_constants
import ecell.emc
import ecell.Session

from sbmlsupport import *


class SBMLImporter:


    def __init__( self, aSBMLDocument=None ):

        self.theSBMLDocument = None
        
        if aSBMLDocument != None:
            self.loadSBML( aSBMLDocument )

    # end of __init__


    def loadSBML( self, aSBMLDocument ):

        if type( aSBMLDocument ) == str:
            self.theSBMLDocument = libsbml.readSBML( aSBMLDocument )
        elif type( aSBMLDocument ) == file:
            self.theSBMLDocument = libsbml.readSBMLFromString( aSBMLDocument.read() )
        else:
            self.theSBMLDocument = aSBMLDocument

        if self.theSBMLDocument.getLevel() != 2:
            self.theSBMLDocument.setLevel( 2 )

        if self.theSBMLDocument.getNumFatals() > 0:
            fatalerrorList = []
            for i in range( self.theSBMLDocument.getNumFatals() ):
                fatalerrorList.append( self.theSBMLDocument.getFatal( i ) )
    
            self.theSBMLDocument = None
            raise SBMLConvertError, \
                  'This SBML document is invalid: %s' \
                  % ( ', '.join( fatalerrorList ) )

    # end of loadSBML
    

    def convertSBMLToEML( self ):

        if not self.theSBMLDocument:
            raise SBMLConvertError, 'No SBML document is defined'

        anEml = ecell.eml.Eml()

        aModelImporter = ModelImporter( self.theSBMLDocument )
        aModelImporter.writeEML( anEml )

        return anEml

    # end of convertSBMLToEML
    

    def createSession( self, aSession=None ):

        if not aSession:
            aSession = ecell.Session( ecell.emc.Simulator() )

        if not self.theSBMLDocument:
            raise SBMLConvertError, 'No SBML document is defined'

        anEml = self.convertSBMLToEML()
        aSession.loadModel( anEml ) # is this supported now?

        return aSession

    # createSession
    

    def saveAsEML( self, filename ):

        anEml = self.convertSBMLToEML()
        anEml.save( filename )

    # end of saveAsEML
    

# end of SBMLImporter


class SBaseImporter:


    def __init__( self, rootobj, sbmlId=None ):
        
        self.initialize( rootobj, sbmlId )

    # end of __init__


    def initialize( self, rootobj, sbmlId=None ):

        self.rootobj = rootobj
        self.theSBase = self.getSBase( sbmlId )

    # end of initialize


    def getSBase( self, sbmlId=None ):

        import inspect
        caller = inspect.getouterframes( inspect.currentframe() )[ 0 ][ 3 ]
        raise NotImplementedError( '%s must be implemented in sub class' % ( caller ) )

    # end of getSBase
    

    def getFullID( self ):

        import inspect
        caller = inspect.getouterframes( inspect.currentframe() )[ 0 ][ 3 ]
        raise NotImplementedError( '%s must be implemented in sub class' % ( caller ) )

    # end of getFullID
    

    def writeEML( self, anEml ):

        import inspect
        caller = inspect.getouterframes( inspect.currentframe() )[ 0 ][ 3 ]
        raise NotImplementedError( '%s must be implemented in sub class' % ( caller ) )

    # end of writeEML
    

# end of SBaseImporter


class ModelImporter( SBaseImporter ):


    def __init__( self, aSBMLDocument ):

        SBaseImporter.__init__( self, aSBMLDocument )

    # end of __init__
    

    def initialize( self, aSBMLDocument, sbmlId=None ):

        SBaseImporter.initialize( self, aSBMLDocument )
        self.theSBMLIdManager = SBMLIdManager( self.rootobj )

    # end of initialize


    def getSBase( self, sbmlId=None ):

        aModel = self.rootobj.getModel()
        if not aModel:
            raise SBMLConvertError, 'Model is not found'
        else:
            return aModel

    # end of getSBase


    def getFullID( self ):

        return self.theSBMLIdManager.getNamespace()

    # end of getFullID
    

    def writeEML( self, anEml ):

        fullIDString = self.getFullID()
        if not anEml.isEntityExist( fullIDString ):
            anEml.createEntity( 'System', fullIDString )
##         else:
##             raise SBMLConvertError, \
##                   'System [%s] is already exist' % ( fullIDString )

        if anEml.getStepperList().count( ODE_STEPPER_ID ) == 0:
            anEml.createStepper( 'ODEStepper', ODE_STEPPER_ID )

        anEml.setEntityProperty( fullIDString, \
                                 'StepperID', [ ODE_STEPPER_ID ] )

        if self.theSBase.isSetName():
            anEml.setEntityProperty( fullIDString, \
                                     'Name', [ self.theSBase.getName() ] )

        for i in range( self.theSBase.getNumCompartments() ):
            aCompartmentImporter = CompartmentImporter( self, i )
            aCompartmentImporter.writeEML( anEml )

        for i in range( self.theSBase.getNumSpecies() ):
            aSpeciesImporter = SpeciesImporter( self, i )
            aSpeciesImporter.writeEML( anEml )

        for i in range( self.theSBase.getNumParameters() ):
            aParameterImporter = ParameterImporter( self, i )        
            aParameterImporter.writeEML( anEml )

        for i in range( self.theSBase.getNumReactions() ):
            aReactionImporter = ReactionImporter( self, i )
            aReactionImporter.writeEML( anEml )

        for i in range( self.theSBase.getNumRules() ):
            aRuleImporter = RuleImporter( self, i )
            aRuleImporter.writeEML( anEml )

        if self.theSBase.getNumEvents() > 0 \
               or self.theSBase.getNumFunctionDefinitions() > 0 \
               or self.theSBase.getNumUnitDefinitions() > 0:
            raise SBMLConvertError, 'Event, FunctionDefinition and UnitDefinition is not supported'
        
    # end of writeEML
    

# end of ModelImporter


class CompartmentImporter( SBaseImporter ):


    def __init__( self, aModelImporter, sbmlId ):
        
        SBaseImporter.__init__( self, aModelImporter, sbmlId )

    # end of __init__


    def getSBase( self, sbmlId ):
        
        aCompartment = self.rootobj.theSBase.getCompartment( sbmlId )
        if not aCompartment:
            raise SBMLConvertError, \
                  'Compartment [%s] is not found' % ( sbmlId )
        elif not aCompartment.isSetId():
            raise SBMLConvertError, \
                  'Compartment [%s] has no id' % ( sbmlId )
        else:
            return aCompartment

    # end of getSBase


    def getFullID( self ):

        sbmlId = self.theSBase.getId()
        return self.rootobj.theSBMLIdManager.getCompartmentFullID( sbmlId )

    # end of getFullID
    

    def createEntity( self, anEml, fullIDString ):

        if anEml.isEntityExist( fullIDString ):
            return
        
        fullID = ecell.ecssupport.createFullID( fullIDString )
        if fullID[ 0 ] != ecell.ecs_constants.SYSTEM:
            return

        if fullID[ 1 ] != '':        
            newSysID = ecell.ecssupport.createFullIDFromSystemPath( fullID[ 1 ] )
            newSysIDString = ecell.ecssupport.createFullIDString( newSysID )
            if not anEml.isEntityExist( newSysIDString ):
                self.createEntity( anEml, newSysIDString )

        anEml.createEntity( 'System', fullIDString )

    # end of createEntity
    

    def writeEML( self, anEml ):

        fullIDString = self.getFullID()
        if not anEml.isEntityExist( fullIDString ):
            # Use not anEml.createEntity but self.createEntity
            self.createEntity( anEml, fullIDString )
##         else:
##             raise SBMLConvertError, \
##                   'System [%s] already exists' % ( fullIDString )

        ## ODEStepper is set as default
        if anEml.getStepperList().count( ODE_STEPPER_ID ) == 0:
            anEml.createStepper( 'ODEStepper', ODE_STEPPER_ID )

        anEml.setEntityProperty( fullIDString, \
                                 'StepperID', [ ODE_STEPPER_ID ] )

        if self.theSBase.isSetSize():
            fullID = ecell.ecssupport.createFullID( fullIDString )
            fullPath = ecell.ecssupport.createSystemPathFromFullID( fullID )
            sizeFullIDString = 'Variable:%s:SIZE' % ( fullPath )
            if not anEml.isEntityExist( sizeFullIDString ):
                anEml.createEntity( 'Variable', sizeFullIDString )
                anEml.setEntityProperty( sizeFullIDString, 'Value', \
                                         [ str( self.theSBase.getSize() ) ] )
##             else:
##                 raise SBMLConvertError, \
##                       'Variable [%s] already exists' % ( sizeFullIDString )

        else:
            fullID = ecell.ecssupport.createFullID( fullIDString )
            fullPath = ecell.ecssupport.createSystemPathFromFullID( fullID )
            sizeFullIDString = 'Variable:%s:SIZE' % ( fullPath )
            if not anEml.isEntityExist( sizeFullIDString ):
                anEml.createEntity( 'Variable', sizeFullIDString )
                anEml.setEntityProperty( sizeFullIDString, 'Value', [ '1.0' ] )

        if self.theSBase.isSetName():
            anEml.setEntityProperty( fullIDString, \
                                     'Name', [ self.theSBase.getName() ] )

    # end of writeEML


# end of CompartmentImporter


class SpeciesImporter( SBaseImporter ):


    def __init__( self, aModelImporter, sbmlId ):

        SBaseImporter.__init__( self, aModelImporter, sbmlId )

    # end of __init__


    def getSBase( self, sbmlId ):

        aSpecies = self.rootobj.theSBase.getSpecies( sbmlId )
        if not aSpecies:
            raise SBMLConvertError, 'Species [%s] is not found' % ( sbmlId )
        elif not aSpecies.isSetId():
            raise SBMLConvertError, 'Species [%s] has no id' % ( sbmlId )
        elif not aSpecies.isSetCompartment():
            raise SBMLConvertError, \
                  'Species [%s] has no compartment' % ( sbmlId )
        else:
            return aSpecies

    # end of getSBase
    

    def getFullID( self ):

        sbmlId = self.theSBase.getId()
        return self.rootobj.theSBMLIdManager.getSpeciesFullID( sbmlId )

    # end of getFullID
    

    def writeEML( self, anEml ):

        fullIDString = self.getFullID()
        if not anEml.isEntityExist( fullIDString ):
            anEml.createEntity( 'Variable', fullIDString )
##         else:
##             raise SBMLConvertError, \
##                   'Variable [%s] already exists' % ( fullIDString )
            
        if self.theSBase.isSetInitialAmount():
            initialAmount = self.theSBase.getInitialAmount()
            anEml.setEntityProperty( fullIDString, \
                                     'Value', [ str( initialAmount ) ] )

        elif self.theSBase.isSetInitialConcentration():
            compartment = self.theSBase.getCompartment()
            aCompartment = self.rootobj.theSBase.getCompartment( compartment )
            if not aCompartment:
                raise SBMLConvertError, \
                      'Compartment [%s] is not found' % ( compartment )

            initialAmount = self.theSBase.getInitialConcentration() \
                            * aCompartment.getSize()
            anEml.setEntityProperty( fullIDString, \
                                     'Value', [ str( initialAmount ) ] )

##         else:
##             raise SBMLConvertError, 'Species [%s] has neither initialAmount nor initialConcentration' % ( self.theSBase.getId() )

        if self.theSBase.isSetName():
            anEml.setEntityProperty( fullIDString, \
                                     'Name', [ self.theSBase.getName() ] )

        if self.theSBase.getBoundaryCondition() == True:
            anEml.setEntityProperty( fullIDString, 'Fixed', [ '1' ] )

    # end of writeEML


# end of SpeciesImporter


class ParameterImporter( SBaseImporter ):


    def __init__( self, aModelImporter, sbmlId ):
        
        SBaseImporter.__init__( self, aModelImporter, sbmlId )

    # end of __init__


    def getSBase( self, sbmlId ):
        
        aParameter = self.rootobj.theSBase.getParameter( sbmlId )
        if not aParameter:
            raise SBMLConvertError, 'Parameter [%s] is not found' % ( sbmlId )
        elif not aParameter.isSetId():
            raise SBMLConvertError, 'Parameter [%s] has no id' % ( sbmlId )
        else:
            return aParameter

    # end of getSBase


    def getFullID( self ):

        sbmlId = self.theSBase.getId()
        return self.rootobj.theSBMLIdManager.getParameterFullID( sbmlId )

    # end of getFullID
    

    def writeEML( self, anEml ):

        fullIDString = self.getFullID()        
        if not anEml.isEntityExist( fullIDString ):
            anEml.createEntity( 'Variable', fullIDString )
##         else:
##             raise SBMLConvertError, \
##                   'Variable [%s] is already exist' % ( fullIDString )
        
        if self.theSBase.isSetValue():
            anEml.setEntityProperty( fullIDString, 'Value', \
                                     [ str( self.theSBase.getValue() ) ] )
##         else:
##             raise SBMLConvertError, \
##                   'Parameter [%s] has no value' % ( self.theSBase.getId() )

        if self.theSBase.isSetName():
            anEml.setEntityProperty( fullIDString, \
                                     'Name', [ self.theSBase.getName() ] )

    # end of writeEML
    

# end of ParameterImporter


class RuleImporter( SBaseImporter ):
    '''
    AlgebraicRule, AssignmentRule and RateRule are integrated into
    RuleImporter. RuleImporter does not have child class for different
    type of Rules.
    '''


    def __init__( self, aModelImporter, sbmlId ):
        
        SBaseImporter.__init__( self, aModelImporter, sbmlId )

    # end of __init__


    def initialize( self, aModelImporter, sbmlId ):

        SBaseImporter.initialize( self, aModelImporter, sbmlId )

        self.sbmlId = sbmlId
        
        self.theExpression = None
        self.theVariableReferenceList = None

    # end of initialize


    def getSBase( self, sbmlId ):
        
        aRule = self.rootobj.theSBase.getRule( sbmlId )
        if not aRule:
            raise SBMLConvertError, 'Rule [%s] not found' % ( sbmlId )
        if not aRule.isSetMath():
            raise SBMLConvertError, 'Rule [%s] has no math' % ( sbmlId )

        # SBML_ASSIGNMENT_RULE or SBML_RATE_RULE
##         if aRule.getType() == 1 or aRule.getType() == 2:
        if type( aRule ) == libsbml.AssignmentRulePtr \
               or type( aRule ) == libsbml.RateRulePtr:
            if not aRule.isSetVariable():
                raise SBMLConvertError, \
                      'Rule [%d] has no variable' % ( sbmlId )
            
        return aRule

    # end of getSBase


    def getFullID( self ):

        return self.rootobj.theSBMLIdManager.getRuleFullID( self.sbmlId )

    # end of getFullID
    

    def writeEML( self, anEml ):

        fullIDString = self.getFullID()
        if not anEml.isEntityExist( fullIDString ):

##             ruleType = self.theSBase.getType()
##             if ruleType == 1: # SBML_ASSIGNMENT_RULE
            if type( self.theSBase ) == libsbml.AssignmentRulePtr:
                if anEml.getStepperList().count( PASSIVE_STEPPER_ID ) == 0:
                    anEml.createStepper( 'PassiveStepper', PASSIVE_STEPPER_ID )

                anEml.createEntity( 'ExpressionAssignmentProcess', \
                                    fullIDString )
                anEml.setEntityProperty( fullIDString, \
                                         'StepperID', \
                                         [ PASSIVE_STEPPER_ID ] )
            else:
                raise SBMLConvertError, \
                      'Algebraic and Rate Rule are not supported'
##         else:
##             raise SBMLConvertError, \
##                   'Process [%s] is already exist' % ( fullIDString )

        self.initializeVariableReferenceList()
        anEml.setEntityProperty( fullIDString, \
                                 'Expression', [ self.theExpression ] )
        anEml.setEntityProperty( fullIDString, 'VariableReferenceList', \
                                 self.theVariableReferenceList )

    # end of writeEML
    

    def initializeVariableReferenceList( self ):

        self.theVariableReferenceList = []

        formulaString = self.theSBase.getFormula()
        
        anASTNode = libsbml.parseFormula( formulaString )
        self.__convertFormulaToExpression( anASTNode )

        self.theExpression = libsbml.formulaToString( anASTNode )

        # SBML_ASSIGNMENT_RULE or SBML_RATE_RULE
##         if self.theSBase.getType() == 1 \
##                or self.theSBase.getType() == 2:
        if type( self.theSBase ) == libsbml.AssignmentRulePtr \
               or type( self.theSBase ) == libsbml.RateRulePtr:

            variable = self.theSBase.getVariable()
            ( fullIDString, sbaseType ) \
              = self.rootobj.theSBMLIdManager.searchFullIDFromId( variable )

            if sbaseType == libsbml.SBML_SPECIES \
                   or sbaseType == libsbml.SBML_PARAMETER:
                pass
            
            elif sbaseType == libsbml.SBML_COMPARTMENT:
                fullID = ecell.ecssupport.createFullID( fullIDString )
                systemPath = ecell.ecssupport.createSystemPathFromFullID( fullID )
                fullIDString = 'Variable:%s:SIZE' % ( systemPath )

            else:
                raise SBMLConvertError, \
                      'SBase [%s] is not found' % ( variable )

            self.__addVariableReference( fullIDString, 1 )

    # end of initializeVariableReferenceList


    def __addVariableReference( self, fullIDString, stoichiometry=0 ):
        '''
        How should unique name for VariableReference be determined
        from fullID?
        '''

        for i in range( len( self.theVariableReferenceList ) ):
            aVariableReference = self.theVariableReferenceList[ i ]
            if aVariableReference[ 1 ] == fullIDString:

                if aVariableReference[ 2 ] != str( stoichiometry ):
                    
##                     if not aVariableReference[ 2 ] == '0' \
##                            and not stoichiometry == 0:
##                         raise SBMLConvertError, 'Unable to overwrite the stoichiometry for Variable [%s]' % ( fullID )

                    coeff = string.atoi( aVariableReference[ 2 ] )
                    self.theVariableReferenceList[ i ][ 2 ] = str( stoichiometry + coeff )
                    return aVariableReference[ 0 ]

                else:
                    return aVariableReference[ 0 ]

        fullID = ecell.ecssupport.createFullID( fullIDString )
        if not fullID[ 0 ] == ecell.ecssupport.VARIABLE:
            raise SBMLConvertError, \
                  'The type of Entity [%s] must be Variable' % ( fullIDString )

        ( name, sbaseType ) = self.rootobj.theSBMLIdManager.getIdFromFullID( fullIDString )
        if not name:
            raise SBMLConvertError, 'There is no SBase corresponding Entity [%s]' % ( fullIDString )
        
        self.theVariableReferenceList.append( [ name, fullIDString, str( stoichiometry ) ] )

        return name

    # end of __addVariableReference


    def __convertFormulaToExpression( self, anASTNode ):

        numChildren = anASTNode.getNumChildren()

        if numChildren == 2:
            self.__convertFormulaToExpression( anASTNode.getLeftChild() )
            self.__convertFormulaToExpression( anASTNode.getRightChild() )
            return anASTNode

        elif numChildren == 1:
            self.__convertFormulaToExpression( anASTNode.getLeftChild() )
            return anASTNode

        elif numChildren == 0:

            if not anASTNode.isNumber():
                
                name = anASTNode.getName()
                ( fullIDString, sbaseType ) \
                  = self.rootobj.theSBMLIdManager.searchFullIDFromId( name )

                if sbaseType == libsbml.SBML_SPECIES:

                    variableName = self.__addVariableReference( fullIDString )

                    fullID = ecell.ecssupport.createFullID( fullIDString )
                    sizeName = self.__addVariableReference( 'Variable:%s:SIZE' % ( fullID[ 1 ] ) )
                    
                    anASTNode.setType( libsbml.AST_DIVIDE )
                    anASTNode.addChild( libsbml.ASTNode( libsbml.AST_NAME ) )
                    anASTNode.addChild( libsbml.ASTNode( libsbml.AST_NAME ) )
                    anASTNode.getLeftChild().setName( '%s.Value' % ( variableName ) )
                    anASTNode.getRightChild().setName( '%s.Value' % ( sizeName ) )
                    return anASTNode
                    
                elif sbaseType == libsbml.SBML_PARAMETER:

                    variableName = self.__addVariableReference( fullIDString )

                    anASTNode.setName( '%s.Value' % ( variableName ) )
                    return anASTNode

                elif sbaseType == libsbml.SBML_COMPARTMENT:

                    fullID = ecell.ecssupport.createFullID( fullIDString )
                    systemPath = ecell.ecssupport.createSystemPathFromFullID( fullID )
                    fullIDString = 'Variable:%s:SIZE' % ( systemPath )
                    
                    variableName = self.__addVariableReference( fullIDString )

                    anASTNode.setName( '%s.Value' % ( variableName ) )
                    return anASTNode

                else:
                    raise SBMLConvertError, 'SBase [%s] not found' % ( name )

    # end of __convertFormulaToExpression


# end of RuleImporter


class ReactionImporter( SBaseImporter ):


    def __init__( self, aModelImporter, sbmlId ):
        
        SBaseImporter.__init__( self, aModelImporter, sbmlId )

    # end of __init__


    def initialize( self, aModelImporter, sbmlId ):

        SBaseImporter.initialize( self, aModelImporter, sbmlId )

        self.theParameterDict = None
        
        self.theExpression = None
        self.theVariableReferenceList = None

    # end of initialize


    def getSBase( self, sbmlId ):
        
        aReaction = self.rootobj.theSBase.getReaction( sbmlId )
        if not aReaction:
            raise SBMLConvertError, 'Reaction [%s] is not found' % ( sbmlId )
        elif not aReaction.isSetId():
            raise SBMLConvertError, 'Reaction [%s] has no id' % ( sbmlId )
        elif not aReaction.isSetKineticLaw():
            raise SBMLConvertError, \
                  'Reaction [%s] has no kinetic law' % ( sbmlId )
        else:
            return aReaction

    # end of getSBase


    def getFullID( self ):

        sbmlId = self.theSBase.getId()
        return self.rootobj.theSBMLIdManager.getReactionFullID( sbmlId )

    # end of getFullID
    

    def writeEML( self, anEml ):

        fullID = self.getFullID()
        if not anEml.isEntityExist( fullID ):
            anEml.createEntity( 'ExpressionFluxProcess', fullID )
##         else:
##             raise SBMLConvertError, \
##                   'Process [%s] already exists' % ( fullID )
        
        self.initializeVariableReferenceList()    
        anEml.setEntityProperty( fullID, \
                                 'Expression', [ self.theExpression ] )
        anEml.setEntityProperty( fullID, 'VariableReferenceList', \
                                 self.theVariableReferenceList )

        for ( parameterName, value ) in self.theParameterDict.items():
            anEml.setEntityProperty( fullID, parameterName, [ str( value ) ] )

        if self.theSBase.isSetName():
            anEml.setEntityProperty( fullID, \
                                     'Name', [ self.theSBase.getName() ] )

    # end of writeEML


    def initializeVariableReferenceList( self ):

        self.theVariableReferenceList = []

        aKineticLaw = self.theSBase.getKineticLaw()

        self.theParameterDict = {}
        for i in range( aKineticLaw.getNumParameters() ):
            aParameter = aKineticLaw.getParameter( i )
            if not aParameter.isSetId():
                raise SBMLConvertError, \
                      'Reaction [%s] is invalid' % ( self.theSBase.getId() )
            elif not  aParameter.isSetValue():
                raise SBMLConvertError, 'Parameter [%s] of Reaction [%s] has no value' % ( aParameter.getId(), self.theSBase.getId() )

            self.theParameterDict[ aParameter.getId() ] = aParameter.getValue()
        
        formulaString = aKineticLaw.getFormula()
        anASTNode = libsbml.parseFormula( formulaString )
        self.__convertFormulaToExpression( anASTNode )

        for i in range( self.theSBase.getNumReactants() ):
            aReactant = self.theSBase.getReactant( i )

            if not aReactant.isSetSpecies():
                raise SBMLConvertError, 'Species is not defined for Reactant [%d] of Reaction [%s]' % ( i, self.theSBase.getId() )
            
            ( fullIDString, sbaseType ) \
              = self.rootobj.theSBMLIdManager.searchFullIDFromId( aReactant.getSpecies() )
            if not ( sbaseType == libsbml.SBML_SPECIES \
                     or sbaseType == libsbml.SBML_PARAMETER \
                     or sbaseType == libsbml.SBML_COMPARTMENT ):
                raise SBMLConvertError, \
                      'SBase [%s] not found' % ( aReactant.getSpecies() )

            if aReactant.isSetStoichiometryMath():
                raise SBMLConvertError, 'Stoichiometry Math is not supported. Check Reactant [%d] of Reaction [%s]' % ( i, self.theSBase.getId() )
            
            stoichiometry = aReactant.getStoichiometry()
            if stoichiometry != int( stoichiometry ):
                raise SBMLConvertError, 'Stoichiometry must be integer. Check Reactant [%d] of Reaction [%s]' % ( i, self.theSBase.getId() )
            
            variableName = self.__addVariableReference( fullIDString, \
                                                        -int( stoichiometry ) )
            
        for i in range( self.theSBase.getNumProducts() ):
            aProduct = self.theSBase.getProduct( i )

            if not aProduct.isSetSpecies():
                raise SBMLConvertError, 'Species is not defined for Product [%d] of Reaction [%s]' % ( i, self.theSBase.getId() )

            ( fullIDString, sbaseType ) \
              = self.rootobj.theSBMLIdManager.searchFullIDFromId( aProduct.getSpecies() )
            if not ( sbaseType == libsbml.SBML_SPECIES \
                     or sbaseType == libsbml.SBML_PARAMETER \
                     or sbaseType == libsbml.SBML_COMPARTMENT ):
                raise SBMLConvertError, \
                      'SBase [%s] not found' % ( aProduct.getSpecies() )

            if aProduct.isSetStoichiometryMath():
                raise SBMLConvertError, 'Stoichiometry Math is not supported. Check Product [%d] of Reaction [%s]' % ( i, self.theSBase.getId() )
            
            stoichiometry = aProduct.getStoichiometry()
            if stoichiometry != int( stoichiometry ):
                raise SBMLConvertError, 'Stoichiometry must be integer. Check Product [%d] of Reaction [%s]' % ( i, self.theSBase.getId() )
            
            variableName = self.__addVariableReference( fullIDString, \
                                                        int( stoichiometry ) )

        for i in range( self.theSBase.getNumModifiers() ):

            aModifier = self.theSBase.getModifier( i )

            if not aModifier.isSetSpecies():
                raise SBMLConvertError, 'Species is not defined for Modifier [%d] of Reaction [%s]' % ( i, self.theSBase.getId() )

            ( fullIDString, sbaseType ) \
              = self.rootobj.theSBMLIdManager.searchFullIDFromId( aModifier.getSpecies() )
            if not ( sbaseType == libsbml.SBML_SPECIES \
                     or sbaseType == libsbml.SBML_PARAMETER \
                     or sbaseType == libsbml.SBML_COMPARTMENT ):
                raise SBMLConvertError, \
                      'SBase [%s] not found' % ( aModifier.getSpecies() )

            variableName = self.__addVariableReference( fullIDString )

        self.theExpression = libsbml.formulaToString( anASTNode )

    # end of initializeVariableReferenceList


    def __addVariableReference( self, fullIDString, stoichiometry=0 ):
        '''
        How should unique name for VariableReference be determined
        from fullID?
        '''

        for i in range( len( self.theVariableReferenceList ) ):
            aVariableReference = self.theVariableReferenceList[ i ]
            if aVariableReference[ 1 ] == fullIDString:

                if aVariableReference[ 2 ] != str( stoichiometry ):
                    
##                     if not aVariableReference[ 2 ] == '0' \
##                            and not stoichiometry == 0:
##                         raise SBMLConvertError, 'Unable to overwrite the stoichiometry for Variable [%s]' % ( fullID )

                    coeff = string.atoi( aVariableReference[ 2 ] )
                    self.theVariableReferenceList[ i ][ 2 ] = str( stoichiometry + coeff )
                    return aVariableReference[ 0 ]

                else:
                    return aVariableReference[ 0 ]

        fullID = ecell.ecssupport.createFullID( fullIDString )
        if not fullID[ 0 ] == ecell.ecssupport.VARIABLE:
            raise SBMLConvertError, \
                  'The type of Entity [%s] must be Variable' % ( fullIDString )

        ( name, sbaseType ) = self.rootobj.theSBMLIdManager.getIdFromFullID( fullIDString )
            
        if not name:
            raise SBMLConvertError, 'There is no SBase corresponding Entity [%s]' % ( fullIDString )
        
        self.theVariableReferenceList.append( [ name, fullIDString, str( stoichiometry ) ] )

        return name

    # end of __addVariableReference


    def __convertFormulaToExpression( self, anASTNode ):

        numChildren = anASTNode.getNumChildren()

        if numChildren == 2:
            self.__convertFormulaToExpression( anASTNode.getLeftChild() )
            self.__convertFormulaToExpression( anASTNode.getRightChild() )
            return anASTNode

        elif numChildren == 1:
            self.__convertFormulaToExpression( anASTNode.getLeftChild() )
            return anASTNode

        elif numChildren == 0:

            if anASTNode.isNumber():
                return anASTNode

            else:
                name = anASTNode.getName()
                ( fullIDString, sbaseType ) \
                  = self.rootobj.theSBMLIdManager.searchFullIDFromId( name )

                if sbaseType == libsbml.SBML_SPECIES:

                    variableName = self.__addVariableReference( fullIDString )

                    fullID = ecell.ecssupport.createFullID( fullIDString )
                    sizeName = self.__addVariableReference( 'Variable:%s:SIZE' % ( fullID[ 1 ] ) )
                    
                    anASTNode.setType( libsbml.AST_DIVIDE )
                    anASTNode.addChild( libsbml.ASTNode( libsbml.AST_NAME ) )
                    anASTNode.addChild( libsbml.ASTNode( libsbml.AST_NAME ) )
                    anASTNode.getLeftChild().setName( '%s.Value' % ( variableName ) )
                    anASTNode.getRightChild().setName( '%s.Value' % ( sizeName ) )
                    return anASTNode
                    
                elif sbaseType == libsbml.SBML_PARAMETER:

                    variableName = self.__addVariableReference( fullIDString )

                    anASTNode.setName( '%s.Value' % ( variableName ) )
                    return anASTNode

                elif sbaseType == libsbml.SBML_COMPARTMENT:

                    fullID = ecell.ecssupport.createFullID( fullIDString )
                    systemPath = ecell.ecssupport.createSystemPathFromFullID( fullID )
                    fullIDString = 'Variable:%s:SIZE' % ( systemPath )
                    
                    variableName = self.__addVariableReference( fullIDString )

                    anASTNode.setName( '%s.Value' % ( variableName ) )
                    return anASTNode

                elif self.theParameterDict.has_key( name ):
                    return anASTNode

                else:
                    raise SBMLConvertError, 'Reaction [%s] has no parameter [%s]' % ( self.theSBase.getId(), name )
                
    # end of __convertFormulaToExpression


# end of ReactionImporter


if __name__ == '__main__':

    import SBMLImporter
    import sys
    import os.path


    def main( filename ):

        ( basenameString, extString ) \
          = os.path.splitext( os.path.basename( filename ) )

        aSBMLImporter = SBMLImporter.SBMLImporter( filename )
        aSBMLImporter.saveAsEML( '%s.eml' % ( basenameString ) )

    # end of main


    if len( sys.argv ) > 1:
        main( sys.argv[ 1 ] )
