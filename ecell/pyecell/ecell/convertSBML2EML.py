#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2008 Keio University
#       Copyright (C) 2005-2008 The Molecular Sciences Institute
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
import string
import re
import os
import time
import getopt
import types

from ecell.eml import *
from convertSBMLFunctions import *
import libsbml


def convertSBML2EML( aSBMLString ):

    aSBMLDocument = libsbml.readSBMLFromString( aSBMLString )
    aSBMLModel = aSBMLDocument.getModel()

    theModel = SBML_Model( aSBMLDocument, aSBMLModel )
    theCompartment = SBML_Compartment( theModel )
    theParameter = SBML_Parameter( theModel )
    theSpecies = SBML_Species( theModel )
    theRule = SBML_Rule( theModel )
    theReaction = SBML_Reaction( theModel )
#    theEvent = SBML_Event( theModel )

    anEml = Eml()

    # ------------------------------
    #  Set Stepper
    # ------------------------------

    anEml.createStepper( 'ODEStepper', 'DE' )
##    anEml.createStepper( 'ODE45Stepper', 'DE' )


    # ------------------------------
    #  Set Compartment ( System )
    # ------------------------------

    # setFullID
    aSystemFullID='System::/'
    anEml.createEntity( 'System', aSystemFullID )
    anEml.setEntityProperty( aSystemFullID, 'StepperID', ['DE'] )
    anEml.setEntityProperty( aSystemFullID, 'Name', ['Default'] )



    for aCompartment in ( theModel.CompartmentList ):

        # initialize
        theCompartment.initialize( aCompartment )

        # getPath
        if ( theModel.Level == 1 ):
            aPath = theModel.getPath( aCompartment[1] )
        elif ( theModel.Level == 2 ):
            aPath = theModel.getPath( aCompartment[0] )           
        
        # setFullID
        if( aPath == '/' ):
            aSystemFullID = 'System::/'
        else:
            aSystemFullID = theCompartment.getCompartmentID( aCompartment )
            anEml.createEntity( 'System', aSystemFullID )


            # setStepper 
            anEml.setEntityProperty( aSystemFullID, 'StepperID', ['DE'] )


        # setName( default = [] )
        if ( theModel.Level == 2 ):
            if ( aCompartment[1] != '' ):
                anEml.setEntityProperty( aSystemFullID,
                                         'Name',
                                         aCompartment[1:2] )

        # setDimensions( default = 3 )
        aDimensionsFullID = 'Variable:' + aPath + ':Dimensions'
        anEml.createEntity( 'Variable', aDimensionsFullID )
        aTmpList = [ str( aCompartment[2] ) ]
        anEml.setEntityProperty( aDimensionsFullID, 'Value', aTmpList[0:1] )

                  
        # setSIZE
        aSizeFullID = 'Variable:' + aPath + ':SIZE'
        anEml.createEntity( 'Variable', aSizeFullID )

        aSizeValue = theCompartment.getCompartmentSize( aCompartment )
        aSizeUnit = theCompartment.getCompartmentUnit( aCompartment )

        # convert to ECELL Unit
        if ( aSizeUnit != '' ):
            aSizeValue = theModel.convertUnit( aSizeUnit, aSizeValue )

        aTmpList = [ str( aSizeValue ) ]
        anEml.setEntityProperty( aSizeFullID, 'Value', aTmpList[0:1] )


        # setConstant( default = 1 )
        if ( aCompartment[7] == 1 ):
            anEml.setEntityProperty( aSizeFullID, 'Fixed', ['1',] )
           
           
    # ------------------------------
    #  Set GlobalParameter ( Variable )
    # ------------------------------

    if ( theModel.ParameterList != [] ):
    
        # setGlobalParameterSystem
        aSystemFullID='System:/:SBMLParameter'
        anEml.createEntity( 'System', aSystemFullID )
        anEml.setEntityProperty( aSystemFullID, 'StepperID', ['DE'] )
        anEml.setEntityProperty( aSystemFullID, 'Name', ['Global Parameter'] )


    for aParameter in theModel.ParameterList:

        # setFullID
        aSystemFullID = theParameter.getParameterID( aParameter )
        anEml.createEntity( 'Variable', aSystemFullID )
            
        # setName
        if ( aParameter[1] != '' ):
            anEml.setEntityProperty( aSystemFullID, 'Name', aParameter[1:2] )

        # setValue
        aTmpList = [ str( theParameter.getParameterValue( aParameter ) ) ]
        anEml.setEntityProperty( aSystemFullID, 'Value', aTmpList[0:1] )


        # setFixed ( default = 1 )
        if ( aParameter[4] == 1 ):
            # aTmpList = [ str( aParameter[4] ) ]
            aTmpList = [ '1' ]
            anEml.setEntityProperty( aSystemFullID, 'Fixed', aTmpList[0:1] )


    # ------------------------------
    #  Set Species ( Variable )
    # ------------------------------

    for aSpecies in theModel.SpeciesList:
        
        ### setFullID ###
        
        aSystemFullID = theSpecies.getSpeciesID( aSpecies )
        anEml.createEntity( 'Variable', aSystemFullID )


        ### setName ###
        
	if( theModel.Level == 2 ):

            if ( aSpecies[1] != '' ):
                anEml.setEntityProperty( aSystemFullID, 'Name', aSpecies[1:2] )


        ### setValue ###
        
        aTmpList = [ str( theSpecies.getSpeciesValue( aSpecies ) ) ]
        anEml.setEntityProperty( aSystemFullID, 'Value', aTmpList[0:1] )


        ### setFixed ###

        aConstant = theSpecies.getConstant( aSpecies )
        anEml.setEntityProperty( aSystemFullID,
                                 'Fixed',
                                 [ str( aConstant ) ] )


    # ------------------------------
    #  Set Rule ( Process )
    # ------------------------------

    if ( theModel.RuleList != [] ):

        ### make Rule System ###
        
        aSystemFullID='System:/:SBMLRule'
        anEml.createEntity( 'System', aSystemFullID )
        anEml.setEntityProperty( aSystemFullID,
                                 'Name',
                                 ['System for SBML Rule'] )

        anEml.setEntityProperty( aSystemFullID, 'StepperID', ['DE'] )

        
    for aRule in theModel.RuleList:

        theRule.initialize()

        ### setFullID ###        
        aSystemFullID = theRule.getRuleID()


        ### Algebraic Rule ###
        if ( aRule[0] == libsbml.SBML_ALGEBRAIC_RULE ):

            anEml.createEntity( 'ExpressionAlgebraicProcess', aSystemFullID )


        ### Assignment Rule ###
        elif ( aRule[0] == libsbml.SBML_ASSIGNMENT_RULE or
               aRule[0] == libsbml.SBML_SPECIES_CONCENTRATION_RULE or
               aRule[0] == libsbml.SBML_COMPARTMENT_VOLUME_RULE or
               aRule[0] == libsbml.SBML_PARAMETER_RULE ):

            anEml.createEntity( 'ExpressionAssignmentProcess', aSystemFullID )

            aVariableType = theRule.getVariableType( aRule[2] )

            if ( aVariableType == libsbml.SBML_SPECIES ):
                theRule.setSpeciesToVariableReference( aRule[2], '1' )
            elif ( aVariableType == libsbml.SBML_PARAMETER ):
                theRule.setParameterToVariableReference( aRule[2], '1' )
            elif ( aVariableType == libsbml.SBML_COMPARTMENT ):
                theRule.setCompartmentToVariableReference( aRule[2], '1' )
            else:
                raise TypeError,\
            "Variable type must be Species, Parameter, or Compartment"

        ### Rate Rule ###
        elif ( aRule[0] == libsbml.SBML_RATE_RULE ):

            anEml.createEntity( 'ExpressionFluxProcess', aSystemFullID )

            aVariableType = theRule.getVariableType( aRule[2] )

            if ( aVariableType == libsbml.SBML_SPECIES ):
                theRule.setSpeciesToVariableReference( aRule[2], '1' )
            elif ( aVariableType == libsbml.SBML_PARAMETER ):
                theRule.setParameterToVariableReference( aRule[2], '1' )
            elif ( aVariableType == libsbml.SBML_COMPARTMENT ):
                theRule.setCompartmentToVariableReference( aRule[2], '1' )
            else:
                raise TypeError,\
            "Variable type must be Species, Parameter, or Compartment"


        else:
            raise TypeError,\
        " The type of Rule must be Algebraic, Assignment or Rate Rule"

        # convert SBML formula  to E-Cell formula
        convertedFormula = [ str( theRule.convertRuleFormula( aRule[1] ) ) ]

        # set Expression Property
        anEml.setEntityProperty( aSystemFullID,
                                 'Expression',
                                 convertedFormula )
        
        # setVariableReferenceList
        anEml.setEntityProperty( aSystemFullID,
                                 'VariableReferenceList',
                                 theRule.VariableReferenceList )



    # ------------------------------
    #  Set Reaction ( Process )
    # ------------------------------

    for aReaction in theModel.ReactionList:

        theReaction.initialize()

        # setFullID
        aSystemFullID = theReaction.getReactionID( aReaction )
        anEml.createEntity( 'ExpressionFluxProcess', aSystemFullID )

        # setName
        if ( theModel.Level == 2 ):
            if( aReaction[1] != '' ):
                anEml.setEntityProperty( aSystemFullID, 'Name', aReaction[1:2] )

        # setSubstrate
        for aSubstrate in aReaction[5]:
            aSubstrateList = []
            aSubstrateList.append( 'S' + str( theReaction.SubstrateNumber ) )
            theReaction.SubstrateNumber = theReaction.SubstrateNumber + 1
            aSubstrateID = theModel.getSpeciesReferenceID( aSubstrate[0] )
            if ( aSubstrateID == None ):
                raise NameError,"Species "+aSubstrate[0]+" not found"

            aSubstrateList.append( 'Variable:' + aSubstrateID )
            if ( aSubstrate[2] != 1 ):
                raise Exception,"Stoichiometry Error : E-Cell System can't set a floating Stoichiometry"
 
            aSubstrateList.append( str( -1 * theReaction.getStoichiometry(
                aSubstrate[0], aSubstrate[1] ) ) )
            theReaction.VariableReferenceList.append( aSubstrateList )

        # setProduct
        for aProduct in aReaction[6]:
            aProductList = []
            aProductList.append( 'P' + str(theReaction.ProductNumber) )
            theReaction.ProductNumber = theReaction.ProductNumber + 1
            aProductID = theModel.getSpeciesReferenceID( aProduct[0] )
            if ( aProductID == None ):
                raise NameError,"Species "+aProduct[0]+" not found"
            
            aProductList.append( 'Variable:' + aProductID )
            if ( aProduct[2] != 1 ):
                raise Exception,"Stoichiometry Error : E-Cell System can't set a floating Stoichiometry"

            aProductList.append( str( 1 * theReaction.getStoichiometry(
                aProduct[0],  aProduct[1] ) ) )

            theReaction.VariableReferenceList.append( aProductList )

        # setCatalyst
        for aModifier in aReaction[7]:
            aModifierList = []
            aModifierList.append( 'C' + str( theReaction.ModifierNumber ) )
            theReaction.ModifierNumber = theReaction.ModifierNumber + 1
            aModifierID = theModel.getSpeciesReferenceID( aModifier )
            if ( aModifierID == None ):
                raise NameError,"Species "+aModifier[0]+" not found"

            aModifierList.append( 'Variable:' + aModifierID )
            aModifierList.append( '0' )
            theReaction.VariableReferenceList.append( aModifierList )


        # setProperty
        if( aReaction[2] != [] ):
            if( aReaction[2][4] != [] ):
                for aParameter in aReaction[2][4]:
                    if ( aParameter[2] != '' ): 
                        aTmpList = [ str( aParameter[2] ) ]
                        if ( theModel.Level == 1 ):
                            anEml.setEntityProperty\
                            ( aSystemFullID, aParameter[1], aTmpList[0:1] )
                        elif ( theModel.Level == 2 ):
                            anEml.setEntityProperty\
                            ( aSystemFullID, aParameter[0], aTmpList[0:1] )
                            
                          
            # --------------------------
            # set "Expression" Property
            # --------------------------
        
            # convert SBML format formula to E-Cell format formula
            if( aReaction[2][0] != '' ):
                anExpression =\
                [ str( theReaction.convertKineticLawFormula( aReaction[2][0] ) ) ]



                # set Expression Property for ExpressionFluxProcess
                anEml.setEntityProperty( aSystemFullID,
                                         'Expression',
                                         anExpression )

                # setVariableReferenceList
                anEml.setEntityProperty( aSystemFullID,
                                         'VariableReferenceList',
                                         theReaction.VariableReferenceList )

    return anEml
