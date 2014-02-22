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

import sys
import re
import os
import time
import getopt
import types

from ecell.eml import *
from convertSBMLFunctions import *
import libsbml


def convertSBML2EML( aSBMLString,
                     anODEStepper     = "ODEStepper",
                     aFixedODEStepper = 'FixedODE1Stepper',
                     aDiscreteStepper = 'DiscreteTimeStepper' ):   ## 'PassiveStepper'

    theDefaultStepperID         = 'Default'
    theDefaultODEStepperID      = 'ODE'
    theDefaultDiscreteStepperID = 'Discrete'

    Steppers = dict(
        ODE      = anODEStepper,
        FixedODE = aFixedODEStepper,
        Discrete = aDiscreteStepper )

    StepperIDs = dict(
        default  = theDefaultStepperID,
        ODE      = theDefaultODEStepperID,
        Discrete = theDefaultDiscreteStepperID )


    aSBMLDocument = libsbml.readSBMLFromString( aSBMLString )

    # ------------------------------
    #  Function Definition Converter
    # ------------------------------

    theFunctionDefinitionConverter = libsbml.SBMLFunctionDefinitionConverter()
    theFunctionDefinitionConverter.setDocument( aSBMLDocument )
    print "    FunctionDefinitionConverter: %s" % \
        libsbml.OperationReturnValue_toString( theFunctionDefinitionConverter.convert() )

    # ------------------------------
    #  Initial Assignment Converter
    # ------------------------------

    theInitialAssignmentConverter = libsbml.SBMLInitialAssignmentConverter()
    theInitialAssignmentConverter.setDocument( aSBMLDocument )
    print "    InitialAssignmentConverter:  %s" % \
        libsbml.OperationReturnValue_toString( theInitialAssignmentConverter.convert() )

    ## FIXME: theInitialAssignmentConverter.convert() always returns None.

    
    aSBMLModel = aSBMLDocument.getModel()

    theModel       = SBML_Model( aSBMLDocument, aSBMLModel )
    theCompartment = SBML_Compartment( theModel )
    theParameter   = SBML_Parameter( theModel )
    theSpecies     = SBML_Species( theModel )
    theRule        = SBML_Rule( theModel )
    theReaction    = SBML_Reaction( theModel )
    theEvent       = SBML_Event( theModel )

##    # dump FunctionDefinition
##    print "\n"
##    for aFunctionDefinition in theModel.FunctionDefinitionList:
##        print "FunctionDefinition: %s\n" % str( aFunctionDefinition )

    anEml = Eml()

    setStepper( theModel, anEml, Steppers, StepperIDs )
    setCompartment( theCompartment, anEml, StepperIDs )
    setParameter( theParameter, anEml, StepperIDs )
    setSpecies( theSpecies, anEml )
    setRule( theRule, anEml, StepperIDs )
    setReaction( theReaction, anEml, StepperIDs )
    setEvent( theEvent, anEml, StepperIDs )

    return anEml



# ------------------------------
#   Set Stepper
# ------------------------------

def setStepper( theModel, anEml, Steppers, StepperIDs ):

    if theModel.isApplicableVariableTimeStep():
        anEml.createStepper( Steppers[ 'ODE' ], StepperIDs[ 'default' ] )
    else:
        anEml.createStepper( Steppers[ 'FixedODE' ], StepperIDs[ 'default' ] )
##        anEml.createStepper( Steppers[ 'Discrete' ], StepperIDs[ 'default' ] )


# ------------------------------
#   Set Compartment ( System )
# ------------------------------

def setCompartment( theCompartment, anEml, StepperIDs ):

    createRootSystem( anEml, StepperIDs[ 'default' ] )

    for aCompartment in ( theCompartment.Model.CompartmentList ):

##        print "Compartment: " + str( aCompartment )

        # initialize
        theCompartment.initialize( aCompartment )

        # getPath
        aPath = theCompartment.Model.getPath( aCompartment[ theCompartment.Model.getKey()[ 'ID' ] ] )
        
        # setFullID
        if( aPath == '/' ):
            aSystemFullID = 'System::/'
        else:
            aSystemFullID = theCompartment.getCompartmentID( aCompartment )
            anEml.createEntity( 'System', aSystemFullID )

            # setStepper 
            anEml.setEntityProperty( aSystemFullID, 'StepperID', [ StepperIDs[ 'default' ] ] )

        # setName( default = [] )
        if ( theCompartment.Model.Level >= 2 ):
            if ( aCompartment[ 'Name' ] != '' ):
                anEml.setEntityProperty( aSystemFullID,
                                         'Name',
                                         [ aCompartment[ 'Name' ] ] )

        # setDimensions( default = 3 )
        aDimensionsFullID = 'Variable:' + aPath + ':Dimensions'
        anEml.createEntity( 'Variable', aDimensionsFullID )
        aTmpList = [ str( aCompartment[ 'SpatialDimension' ] ) ]
        anEml.setEntityProperty( aDimensionsFullID, 'Value', aTmpList[0:1] )

        # setSIZE
        aSizeFullID = 'Variable:' + aPath + ':SIZE'
        anEml.createEntity( 'Variable', aSizeFullID )

        aSizeValue = theCompartment.getCompartmentSize( aCompartment )
        aSizeUnit = theCompartment.getCompartmentUnit( aCompartment )

        # convert to ECELL Unit
        if ( aSizeUnit != '' ):
            aSizeValue = theCompartment.Model.convertUnit( aSizeUnit, aSizeValue, theCompartment.Model )

        anEml.setEntityProperty( aSizeFullID, 'Value', [ str( aSizeValue ) ] )

        # setConstant( default = 1 )
        if ( aCompartment[ 'Constant' ] == 1 ):
            anEml.setEntityProperty( aSizeFullID, 'Fixed', [ '1' ] )


# ----------------------------------
#   Set GlobalParameter ( Variable )
# ----------------------------------

def setParameter( theParameter, anEml, StepperIDs ):

    if ( theParameter.Model.ParameterList != [] ):
    
        # setGlobalParameterSystem
        theParameterSystemFullID = theParameter.getSystemFullID()
        anEml.createEntity( 'System', theParameterSystemFullID )
        anEml.setEntityProperty( theParameterSystemFullID,
                                 'StepperID',
                                 [ StepperIDs[ 'default' ] ] )
        anEml.setEntityProperty( theParameterSystemFullID,
                                 'Name',
                                 [ 'Global Parameter' ] )


    for aParameter in theParameter.Model.ParameterList:

        # setFullID
        aSystemFullID = theParameter.generateFullID( aParameter )
        anEml.createEntity( 'Variable', aSystemFullID )
            
        # setName
        if ( aParameter[ 'Name' ] != '' ):
            anEml.setEntityProperty( aSystemFullID,
                                     'Name',
                                     [ aParameter[ 'Name' ] ] )

        # setValue
        anEml.setEntityProperty( aSystemFullID,
                                 'Value',
                                 [ str( theParameter.getParameterValue( aParameter ) ) ] )

        # setFixed ( default = 1 )
        if ( aParameter[ 'Constant' ] == 1 ):
            anEml.setEntityProperty( aSystemFullID, 'Fixed', [ '1' ] )


# ------------------------------
#   Set Species ( Variable )
# ------------------------------

def setSpecies( theSpecies, anEml ):

    for aSpecies in theSpecies.Model.SpeciesList:
        
##        print "Species: " + str( aSpecies )
        
        ### set FullID ###
        
        aSystemFullID = theSpecies.generateFullID( aSpecies )
        anEml.createEntity( 'Variable', aSystemFullID )

        ### set Name ###
        
        if( theSpecies.Model.Level >= 2 ):

            if ( aSpecies[ 'Name' ] != '' ):
                anEml.setEntityProperty( aSystemFullID, 'Name', [ aSpecies[ 'Name' ] ] )

        ### set Value ###
        
        aInitialValueDic = theSpecies.getInitialValue( aSpecies )
        anEml.setEntityProperty( aSystemFullID,
                                 aInitialValueDic[ 'Property' ],
                                 [ str( aInitialValueDic[ 'Value' ] ) ] )

        ### set Fixed ###

        if ( theSpecies.isConstant( aSpecies )): aFixedValue = "1"
        else: aFixedValue = "0"
        
        anEml.setEntityProperty( aSystemFullID,
                                 'Fixed',
                                 [ aFixedValue ] )


# ------------------------------
#   Set Rule ( Process )
# ------------------------------

def setRule( theRule, anEml, StepperIDs ):

    if ( theRule.Model.RuleList != [] ):

        ### make Rule System ###
        
        theRuleSystemFullID = theRule.getSystemFullID()
        anEml.createEntity( 'System', theRuleSystemFullID )
        anEml.setEntityProperty( theRuleSystemFullID,
                                 'Name',
                                 ['System for SBML Rule'] )
        anEml.setEntityProperty( theRuleSystemFullID,
                                 'StepperID',
                                 [ StepperIDs[ 'default' ] ] )

    for aRule in theRule.Model.RuleList:

##        print "Rule: " + str( aRule )

        theRule.initialize()

        ### setFullID ###        
        aSystemFullID = theRule.generateFullID( aRule )


        ### Algebraic Rule ###
        if ( aRule[ 'Type' ] == libsbml.SBML_ALGEBRAIC_RULE ):

            anEml.createEntity( 'ExpressionAlgebraicProcess', aSystemFullID )
            anEml.setEntityProperty( aSystemFullID, 'StepperID', [ StepperIDs[ 'default' ] ] )


        ### Assignment Rule ###
        elif aRule[ 'Type' ] in ( libsbml.SBML_ASSIGNMENT_RULE,
                                    libsbml.SBML_SPECIES_CONCENTRATION_RULE,
                                    libsbml.SBML_COMPARTMENT_VOLUME_RULE,
                                    libsbml.SBML_PARAMETER_RULE ):

            anEml.createEntity( 'ExpressionAssignmentProcess', aSystemFullID )
            anEml.setEntityProperty( aSystemFullID, 'StepperID', [ StepperIDs[ 'default' ] ] )

            if( aRule[ 'Type' ] == libsbml.SBML_ASSIGNMENT_RULE ):
                anEml.setEntityProperty( aSystemFullID, 'Name', [ "Assignment rule for '%s'" % aRule[ 'Variable' ] ] )

            theRule.updateVariableReferenceList( aRule[ 'Variable' ], '1' )

        ### Rate Rule ###
        elif ( aRule[ 'Type' ] == libsbml.SBML_RATE_RULE ):

            anEml.createEntity( 'ExpressionFluxProcess', aSystemFullID )
            anEml.setEntityProperty( aSystemFullID, 'Name', [ "Rate rule for '%s'" % aRule[ 'Variable' ] ] )

            theRule.updateVariableReferenceList( aRule[ 'Variable' ], '1' )


        else:
            raise TypeError,\
                " The type of Rule must be Algebraic, Assignment or Rate Rule"

        # convert SBML formula  to E-Cell formula
        convertedFormula = [ str( theRule.convertFormula( aRule[ 'Formula' ] ) ) ]

        # set Expression Property
        anEml.setEntityProperty( aSystemFullID,
                                 'Expression',
                                 convertedFormula )
        
        # setVariableReferenceList
        anEml.setEntityProperty( aSystemFullID,
                                 'VariableReferenceList',
                                 theRule.VariableReferenceList )


# ------------------------------
#   Set Reaction ( Process )
# ------------------------------

def setReaction( theReaction, anEml, StepperIDs ):

    for aReaction in theReaction.Model.ReactionList:

##        print "Reaction: " + str( aReaction )

        theReaction.initialize()

        # setFullID
        aSystemFullID = theReaction.generateFullID( aReaction )
        anEml.createEntity( 'ExpressionFluxProcess', aSystemFullID )

        # setName
        if ( theReaction.Model.Level >= 2 ):
            if( aReaction[ 'Name' ] != '' ):
                anEml.setEntityProperty( aSystemFullID, 'Name', [ aReaction[ 'Name' ] ] )
            else:
                anEml.setEntityProperty( aSystemFullID, 'Name', [ theReaction.getChemicalEquation( aReaction ) ] )
        else:
            anEml.setEntityProperty( aSystemFullID, 'Name', [ theReaction.getChemicalEquation( aReaction ) ] )

        # setSubstrate
        updateVariableReferenceListBySpeciesList( theReaction, aReaction[ 'Reactants' ], -1 )
        # setProduct
        updateVariableReferenceListBySpeciesList( theReaction, aReaction[ 'Products' ] )
        # setCatalyst
        updateVariableReferenceListBySpeciesList( theReaction, aReaction[ 'Modifiers' ] )


        # setProperty
        if( aReaction[ 'KineticLaw' ] != [] ):
            if( aReaction[ 'KineticLaw' ][ 'Parameters' ] != [] ):
                for aParameter in aReaction[ 'KineticLaw' ][ 'Parameters' ]:
                    if ( aParameter[ 'Value' ] != '' ): 
                        anEml.setEntityProperty( aSystemFullID,
                                                 aParameter[ theReaction.Model.getKey()[ 'ID' ] ],
                                                 [ str( aParameter[ 'Value' ] ) ] )


            # --------------------------
            # set "Expression" Property
            # --------------------------
        
            # convert SBML format formula to E-Cell format formula
            if( aReaction[ 'KineticLaw' ][ 'Formula' ] != '' ):
##                print "Kinetic Law: %s" % aReaction[ 'KineticLaw' ]
                anExpression =\
                [ str( theReaction.convertFormula( aReaction[ 'KineticLaw' ][ 'Formula' ], aReaction[ 'KineticLaw' ][ 'Parameters' ] ) ) ]



                # set Expression Property for ExpressionFluxProcess
                anEml.setEntityProperty( aSystemFullID,
                                         'Expression',
                                         anExpression )

                # setVariableReferenceList
                anEml.setEntityProperty( aSystemFullID,
                                         'VariableReferenceList',
                                         theReaction.VariableReferenceList )


# ------------------------------
#   Set Event ( Process )
# ------------------------------

def setEvent( theEvent, anEml, StepperIDs ):

    if ( theEvent.Model.EventList != [] ):

        ### make Event System ###
        
        theEventSystemFullID = theEvent.getSystemFullID()
        anEml.createEntity( 'System', theEventSystemFullID )
        anEml.setEntityProperty( theEventSystemFullID,
                                 'Name',
                                 ['System for SBML Event'] )
        anEml.setEntityProperty( theEventSystemFullID,
                                 'StepperID',
                                 [ StepperIDs[ 'default' ] ] )


    for anEvent in theEvent.Model.EventList:

##        print "Event: " + str( anEvent )

        theEvent.initialize()

        ### setFullID ###        
        aSystemFullID = theEvent.generateFullID( anEvent )

        anEml.createEntity( 'ExpressionEventProcess', aSystemFullID )
        anEml.setEntityProperty( aSystemFullID, 'StepperID', [ StepperIDs[ 'default' ] ] )
        anEml.setEntityProperty( aSystemFullID, 'Name', [ str( theEvent.getEventName( anEvent ) ) ] )

        # convert EventAssignment
        
        theEventAssignmentList = []
        for anEventAssignment in anEvent[ 'EventAssignments' ]:
            
            theEvent.updateVariableReferenceList( anEventAssignment[ 'Variable' ], '1' )
            
            aConvertedEventAssignment = []
            aConvertedEventAssignment.append( anEventAssignment[ 'Variable' ] )
            aConvertedEventAssignment.append( str( theEvent.convertFormula( anEventAssignment[ 'String' ] )))
            theEventAssignmentList.append( aConvertedEventAssignment )

        # convert Trigger
        convertedTrigger = [ str( theEvent.convertFormula( anEvent[ 'Trigger' ] )) ]

        # convert Delay
        if ( anEvent[ 'Delay' ] != '' ):
            convertedDelay = [ str( theEvent.convertFormula( anEvent[ 'Delay' ] )) ]
        else:
            convertedDelay = '0.0'

        # set Expression Property
        anEml.setEntityProperty( aSystemFullID,
                                 'EventAssignmentList',
                                 theEventAssignmentList )
        
        # set Expression Property
        anEml.setEntityProperty( aSystemFullID,
                                 'Trigger',
                                 convertedTrigger )
        
        # set Expression Property
        anEml.setEntityProperty( aSystemFullID,
                                 'Delay',
                                 convertedDelay )
        
        # setVariableReferenceList
        anEml.setEntityProperty( aSystemFullID,
                                 'VariableReferenceList',
                                 theEvent.VariableReferenceList )


def updateVariableReferenceListBySpeciesList( theReaction, aReactingSpeciesList, theDirection = 1 ):
    for aReactingSpecies in aReactingSpeciesList:
        if isinstance( aReactingSpecies, list ):
            _aReactingSpecies = aReactingSpecies
        elif isinstance( aReactingSpecies, str ):
            _aReactingSpecies = [ aReactingSpecies, 0, 1 ]
        else:
            raise Exception,"DEBUG : Unexpected instance during converting Reaction"
        
        if ( _aReactingSpecies[2] != 1 ):     # ReactantDenominator
            raise Exception,"Stoichiometry Error : E-Cell System can't set a floating Stoichiometry"
        
        aStoichiometryInt = theDirection * theReaction.getStoichiometry( _aReactingSpecies[0], _aReactingSpecies[1] )
        
        aCorrespondingVariableReference = theReaction.getVariableReference( _aReactingSpecies[0] )
        if aCorrespondingVariableReference:
            aCorrespondingVariableReference[ 2 ] = str( int( aCorrespondingVariableReference[ 2 ] ) + aStoichiometryInt )
        
        else:
            aReactingSpeciesList = []
            aReactingSpeciesList.append( _aReactingSpecies[0] )
##            theReaction.SubstrateNumber = theReaction.SubstrateNumber + 1
            aVariableFullID = theReaction.Model.getSpeciesReferenceID( _aReactingSpecies[0] )
            if ( aVariableFullID == None ):
                raise NameError,"Species "+ _aReactingSpecies[0] +" not found"

            aReactingSpeciesList.append( 'Variable:' + aVariableFullID )
 
            aReactingSpeciesList.append( str( aStoichiometryInt ) )
            theReaction.VariableReferenceList.append( aReactingSpeciesList )


def createRootSystem( anEml, aStepperID ):
    aSystemFullID='System::/'
    anEml.createEntity( 'System', aSystemFullID )
    anEml.setEntityProperty( aSystemFullID, 'StepperID', [ aStepperID ] )
    anEml.setEntityProperty( aSystemFullID, 'Name', [ 'default' ] )
