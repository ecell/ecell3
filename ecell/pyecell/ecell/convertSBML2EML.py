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

import sys, os
import re
import time
import getopt
import types
import decimal, fractions

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

##    theUndetermined = 

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

    set_Steppers( theModel, anEml, Steppers, StepperIDs )
    set_Compartments( theCompartment, anEml, StepperIDs )
    set_GlobalParameters( theParameter, anEml, StepperIDs )
    set_Species( theSpecies, anEml )
    set_Rules( theRule, anEml, StepperIDs )
    set_Reactions( theReaction, anEml, StepperIDs )
    set_Events( theEvent, anEml, StepperIDs )

    return anEml



# ------------------------------
#   Set Stepper
# ------------------------------

def set_Steppers( theModel, anEml, Steppers, StepperIDs ):

    if theModel.is_applicable_variable_time_step():
        anEml.createStepper( Steppers[ 'ODE' ], StepperIDs[ 'default' ] )
    else:
        anEml.createStepper( Steppers[ 'FixedODE' ], StepperIDs[ 'default' ] )
##        anEml.createStepper( Steppers[ 'Discrete' ], StepperIDs[ 'default' ] )


# ------------------------------
#   Set Compartment ( System )
# ------------------------------

def set_Compartments( theCompartment, anEml, StepperIDs ):

    create_root_System( anEml, StepperIDs[ 'default' ] )

    for aCompartment in ( theCompartment.Model.CompartmentList ):

##        print "Compartment: " + str( aCompartment )

        # initialize
        theCompartment.initialize( aCompartment )

        # getPath
        aPath = theCompartment.Model.get_path( aCompartment[ theCompartment.Model.keys[ 'ID' ] ] )
        
        # setFullID
        if( aPath == '/' ):
            aSystemFullID = 'System::/'
        else:
            aSystemFullID = theCompartment.get_System_FullID( aCompartment )
            anEml.createEntity( 'System', aSystemFullID )

            # set Stepper 
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

        aSizeValue = theCompartment.get_Compartment_size( aCompartment )
        aSizeUnit  = theCompartment.get_Compartment_unit( aCompartment )

        # convert to ECELL Unit
        if ( aSizeUnit != '' ):
            aSizeValue = theCompartment.Model.convert_unit( aSizeUnit, aSizeValue, theCompartment.Model )

        anEml.setEntityProperty( aSizeFullID, 'Value', [ str( aSizeValue ) ] )

        # setConstant( default = 1 )
        if ( aCompartment[ 'Constant' ] == 1 ):
            anEml.setEntityProperty( aSizeFullID, 'Fixed', [ '1' ] )


# ----------------------------------
#   Set GlobalParameter ( Variable )
# ----------------------------------

def set_GlobalParameters( theParameter, anEml, StepperIDs ):

    if ( theParameter.Model.ParameterList != [] ):
    
        # setGlobalParameterSystem
        theParameterSystemFullID = theParameter.get_System_FullID()
        anEml.createEntity( 'System', theParameterSystemFullID )
        anEml.setEntityProperty( theParameterSystemFullID,
                                 'StepperID',
                                 [ StepperIDs[ 'default' ] ] )
        anEml.setEntityProperty( theParameterSystemFullID,
                                 'Name',
                                 [ 'Global Parameter' ] )


    for aParameter in theParameter.Model.ParameterList:

        # setFullID
        aSystemFullID = theParameter.generate_FullID_from_SBML_entity( aParameter )
        anEml.createEntity( 'Variable', aSystemFullID )
            
        # setName
        if ( aParameter[ 'Name' ] != '' ):
            anEml.setEntityProperty( aSystemFullID,
                                     'Name',
                                     [ aParameter[ 'Name' ] ] )

        # setValue
        anEml.setEntityProperty( aSystemFullID,
                                 'Value',
                                 [ str( aParameter[ 'Value' ] ) ] )

        # setFixed ( default = 1 )
        if ( aParameter[ 'Constant' ] == 1 ):
            anEml.setEntityProperty( aSystemFullID, 'Fixed', [ '1' ] )


# ------------------------------
#   Set Species ( Variable )
# ------------------------------

def set_Species( theSpecies, anEml ):

    for aSpecies in theSpecies.Model.SpeciesList:
        
##        print "Species: " + str( aSpecies )
        
        ### set FullID ###
        
        aSystemFullID = theSpecies.generate_FullID_from_SBML_entity( aSpecies )
        anEml.createEntity( 'Variable', aSystemFullID )

        ### set Name ###
        
        if( theSpecies.Model.Level >= 2 ):

            if ( aSpecies[ 'Name' ] != '' ):
                anEml.setEntityProperty( aSystemFullID, 'Name', [ aSpecies[ 'Name' ] ] )

        ### set Value ###
        
        aInitialValueDic = theSpecies.get_initial_value( aSpecies )
        anEml.setEntityProperty( aSystemFullID,
                                 aInitialValueDic[ 'Property' ],
                                 [ str( aInitialValueDic[ 'Value' ] ) ] )

        ### set Fixed ###

        if ( theSpecies.is_constant( aSpecies )): aFixedValue = "1"
        else: aFixedValue = "0"
        
        anEml.setEntityProperty( aSystemFullID,
                                 'Fixed',
                                 [ aFixedValue ] )


# ------------------------------
#   Set Rule ( Process )
# ------------------------------

def set_Rules( theRule, anEml, StepperIDs ):

    if ( theRule.Model.RuleList != [] ):

        ### make Rule System ###
        
        theRuleSystemFullID = theRule.get_System_FullID()
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
        aSystemFullID = theRule.generate_FullID_from_SBML_entity( aRule )


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

            theRule.add_Entity_to_VariableReferenceList( aRule[ 'Variable' ], '1' )

        ### Rate Rule ###
        elif ( aRule[ 'Type' ] == libsbml.SBML_RATE_RULE ):

            anEml.createEntity( 'ExpressionFluxProcess', aSystemFullID )
            anEml.setEntityProperty( aSystemFullID, 'Name', [ "Rate rule for '%s'" % aRule[ 'Variable' ] ] )

            theRule.add_Entity_to_VariableReferenceList( aRule[ 'Variable' ], '1' )


        else:
            raise TypeError,\
                " The type of Rule must be Algebraic, Assignment or Rate Rule"

        # convert SBML formula  to E-Cell formula
        convertedFormula = [ str( theRule.convert_SBML_Formula_to_ecell_Expression( aRule[ 'Math' ] ) ) ]

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

def set_Reactions( theReaction, anEml, StepperIDs ):

    for aReaction in theReaction.Model.ReactionList:

##        print "Reaction: " + str( aReaction )

        theReaction.initialize()

        # setFullID
        aSystemFullID = theReaction.generate_FullID_from_SBML_entity( aReaction )
        anEml.createEntity( 'ExpressionFluxProcess', aSystemFullID )

        # setName
        if ( theReaction.Model.Level >= 2 ):
            if( aReaction[ 'Name' ] != '' ):
                aName = aReaction[ 'Name' ]
            else:
                aName = theReaction.get_chemical_equation( aReaction )
        else:
            aName = theReaction.get_chemical_equation( aReaction )

        if aReaction[ 'CommonDemoninator' ] != 1.0:
            aName = '%s, ( denominatior = %i )' % ( aName, aReaction[ 'CommonDemoninator' ] )

        anEml.setEntityProperty( aSystemFullID, 'Name', [ aName ] )

        update_VariableReferenceList( theReaction, aReaction )

        # setProperty
        if( aReaction[ 'KineticLaw' ] != [] ):
            if( aReaction[ 'KineticLaw' ][ 'Parameters' ] != [] ):
                for aParameter in aReaction[ 'KineticLaw' ][ 'Parameters' ]:
                    if ( aParameter[ 'Value' ] != '' ): 
                        anEml.setEntityProperty( aSystemFullID,
                                                 aParameter[ theReaction.Model.keys[ 'ID' ] ],
                                                 [ str( aParameter[ 'Value' ] ) ] )


            # --------------------------
            # set "Expression" Property
            # --------------------------
        
            # convert SBML format formula to E-Cell format formula
            if( aReaction[ 'KineticLaw' ][ 'Math' ] != None ):
##                print "Kinetic Law: %s" % aReaction[ 'KineticLaw' ]
                anExpression =\
                [ str( theReaction.convert_SBML_Formula_to_ecell_Expression( 
                    aReaction[ 'KineticLaw' ][ 'Math' ],
                    aReaction[ 'KineticLaw' ][ 'Parameters' ],
                    aReaction[ 'CommonDemoninator' ] ) ) ]



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

def set_Events( theEvent, anEml, StepperIDs ):

    if ( theEvent.Model.EventList != [] ):

        ### make Event System ###
        
        theEventSystemFullID = theEvent.get_System_FullID()
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
        aSystemFullID = theEvent.generate_FullID_from_SBML_entity( anEvent )

        anEml.createEntity( 'ExpressionEventProcess', aSystemFullID )
        anEml.setEntityProperty( aSystemFullID, 'StepperID', [ StepperIDs[ 'default' ] ] )
        anEml.setEntityProperty( aSystemFullID, 'Name', [ str( theEvent.get_Event_Name( anEvent ) ) ] )

        # convert EventAssignment
        
        theEventAssignmentList = []
        for anEventAssignment in anEvent[ 'EventAssignments' ]:
            
            theEvent.add_Entity_to_VariableReferenceList( anEventAssignment[ 'Variable' ], '1' )
            
            aConvertedEventAssignment = []
            aConvertedEventAssignment.append( anEventAssignment[ 'Variable' ] )
            aConvertedEventAssignment.append( str( theEvent.convert_SBML_Formula_to_ecell_Expression( anEventAssignment[ 'Math' ] )))
            theEventAssignmentList.append( aConvertedEventAssignment )

        # convert Trigger
        convertedTrigger = [ str( theEvent.convert_SBML_Formula_to_ecell_Expression( anEvent[ 'Trigger' ] )) ]

        # convert Delay
        if ( anEvent[ 'Delay' ] != '' ):
            convertedDelay = [ str( theEvent.convert_SBML_Formula_to_ecell_Expression( anEvent[ 'Delay' ] )) ]
        else:
            convertedDelay = [ '0.0' ]

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


def update_VariableReferenceList( theReaction, aReaction  ):
    # setSubstrate
    _update_VariableReferenceList_by_reactants_list( theReaction,
                                                     aReaction[ theReaction.Model.keys[ 'ID' ] ],
                                                     aReaction[ 'Reactants' ],
                                                     -1,
                                                     aReaction[ 'CommonDemoninator' ] )
    # setProduct
    _update_VariableReferenceList_by_reactants_list( theReaction,
                                                     aReaction[ theReaction.Model.keys[ 'ID' ] ],
                                                     aReaction[ 'Products' ],
                                                     1,
                                                     aReaction[ 'CommonDemoninator' ] )
    # setCatalyst
    _update_VariableReferenceList_by_reactants_list( theReaction,
                                                     aReaction[ theReaction.Model.keys[ 'ID' ] ],
                                                     aReaction[ 'Modifiers' ],
                                                     1,
                                                     aReaction[ 'CommonDemoninator' ] )


def _update_VariableReferenceList_by_reactants_list( theReaction, aReactionID, aReactingSpeciesList, theDirection = 1, denominator = 1 ):
    for aReactingSpecies in aReactingSpeciesList:
        if isinstance( aReactingSpecies, dict ):
            _aReactingSpecies = aReactingSpecies
        elif isinstance( aReactingSpecies, str ):
            _aReactingSpecies = {
                theReaction.Model.keys[ 'ID' ] : aReactingSpecies,
                'Stoichiometry'                : 0,
                'StoichiometryMath'            : None,
                'Denominator'                  : 1 }
        else:
            raise Exception,"DEBUG : Unexpected instance during converting Reaction"
        
        if ( _aReactingSpecies[ 'Denominator' ] != 1 ):     # ReactantDenominator
            raise Exception,"Stoichiometry Error : E-Cell System can't set a floating Stoichiometry"
        
        elif ( _aReactingSpecies[ 'StoichiometryMath' ] != None ):
            raise Exception,"At present, StoichiometryMath is not supported. ( Reaction ID: %s, Reactant ID: %s, StoichiometryMath: %s )" % ( aReactionID, _aReactingSpecies[ theReaction.Model.keys[ 'ID' ] ], _aReactingSpecies[ 'StoichiometryMath' ] )
        
        _aReactingSpeciesEntity = theReaction.Model.get_Entity_by_ID( _aReactingSpecies[ theReaction.Model.keys[ 'ID' ] ] )
        if ( _aReactingSpeciesEntity == False ) or ( _aReactingSpeciesEntity[ 0 ] != libsbml.SBML_SPECIES ):
            raise TypeError,\
                'Species "%s" not found' % _aReactingSpecies[ theReaction.Model.keys[ 'ID' ] ]
        else:
            _aReactingSpeciesEntity = _aReactingSpeciesEntity[ 1 ]

        if _aReactingSpeciesEntity[ 'Constant' ]:
            aStoichiometryInt = 0
            
        else:
##            if _aReactingSpecies[ 'Stoichiometry' ] != int( _aReactingSpecies[ 'Stoichiometry' ] ):
##                raise TypeError, 'Stoichiometry must be integer.'
            
            aStoichiometryInt = theDirection * int( _aReactingSpecies[ 'Stoichiometry' ] * denominator )
        
        aCorrespondingVariableReference = theReaction.get_VariableReference( _aReactingSpecies[ theReaction.Model.keys[ 'ID' ] ] )
        if aCorrespondingVariableReference:
            aCorrespondingVariableReference[ 2 ] = str( int( aCorrespondingVariableReference[ 2 ] ) + aStoichiometryInt )
        
        else:
            aReactingSpeciesList = []
            aReactingSpeciesList.append( _aReactingSpecies[ theReaction.Model.keys[ 'ID' ] ] )
##            theReaction.SubstrateNumber = theReaction.SubstrateNumber + 1
            aVariableFullID = theReaction.Model.get_SpeciesReference_ID( _aReactingSpecies[ theReaction.Model.keys[ 'ID' ] ] )
            if ( aVariableFullID == None ):
                raise NameError,"Species "+ _aReactingSpecies[ theReaction.Model.keys[ 'ID' ] ] +" not found"

            aReactingSpeciesList.append( 'Variable:' + aVariableFullID )
 
            aReactingSpeciesList.append( str( aStoichiometryInt ) )
            theReaction.VariableReferenceList.append( aReactingSpeciesList )


def create_root_System( anEml, aStepperID ):
    aSystemFullID='System::/'
    anEml.createEntity( 'System', aSystemFullID )
    anEml.setEntityProperty( aSystemFullID, 'StepperID', [ aStepperID ] )
    anEml.setEntityProperty( aSystemFullID, 'Name', [ 'default' ] )
