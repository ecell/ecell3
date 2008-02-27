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

import sys
import string
import re
import os
import time
import getopt
import types

from ecell.ecs_constants import *
from ecell.eml import *
import libsbml

__all__ = (
    'convert'
    )


''' Compartment List '''

#   " [[ 0 : Id ,
#        1 : Name ,
#        2 : SpatialDimension ,
#        3 : Size ,
#        4 : Volume ,
#        5 : Unit ,
#        6 : Ouside ,
#        7 : Constant ]] "

    
''' Event List '''

#   " [[ 0 : Id ,
#        1 :  Name ,
#        2 :  StringTrigger ,
#        3 :  StringDelay ,
#        4 :  TimeUnit ,
#        5 :  [[ VariableAssignment , StringAssignment ]] ]] "


''' FunctionDefinition List '''

#   " [[ 0 : Id ,
#        1 : Name ,
#        2 : String ]] "
    

''' Parameter List '''

#   " [[ 0 : Id ,
#        1 : Name ,
#        2 : Value ,
#        3 : Unit ,
#        4 : Constant ]] "


''' Reaction List '''

#   " [[ 0 : Id ,
#        1 : Name ,
#        2 : [ Formula , String , TimeUnit , SubstanceUnit , [[ ParameterId , ParameterName , ParameterValue , ParameterUnit , ParameterConstant ]] , ExpressionAnnotation ] ,
#        3 : Reversible ,
#        4 : Fast ,
#        5 : [[ ReactantSpecies , ( ReactantStoichiometry , ReactantStoichiometryMath ) , ReactantDenominator  ]] ,
#        6 : [[  ProductSpecies , ( ProductStoichiometry , ProductStoichiometryMath ) , ProductDenominator ]] ,
#        7 : [[ ModifierSpecies ]] ]] "

    
''' Rule List '''

#   " [[ 1 : Type,
#        2 : Formula,
#        3 : Variable ]] "


''' Species List '''

#   " [[ 0 : Id ,
#        1 : Name ,
#        2 : Compartment ,
#        3 : InitialAmount ,
#        4 : InitialConcentration ,
#        5 : SubstanceUnit ,
#        6 : SpatialSizeUnit ,
#        7 : Unit ,
#        8 : HasOnlySubstanceUnit ,
#        9 : BoundaryCondition ,
#        10 : Charge ,
#        11 : Constant ]] "
    

''' UnitDefinition List '''

#   " [[ 0 : Id ,
#        1 : Name ,
#        2 : [[ Kind , Exponent , Scale , Multiplier , Offset ]] ]] "

def sub( fun , indata ):
    if indata != []:
        outdata = fun( indata )
        if not outdata:
            outdata = []
    else:
        outdata = []

    return outdata

class Model:
    def __init__( self, aSBMLDocument ):
        self.CompartmentSize = {}
        self.CompartmentUnit = {}
        self.FunctionDefinition = {}

        aSBMLModel = aSBMLDocument.getModel()
        self.theSBMLModel = aSBMLModel
        self.Level = aSBMLDocument.getLevel()
        self.Version = aSBMLDocument.getVersion() 
        self.theCompartmentList = None
        self.theEventList = None
        self.theFunctionDefinitionMap = None
        self.theParameterList = None
        self.theReactionList = None
        self.theRuleList = None
        self.theSpeciesList = None
        self.theUnitDefinitionList = None

    def getPath( self, aCompartmentID ):
        if aCompartmentID == 'default':
            return '/'
        
        if self.Level == 1:
            for aCompartment in self.getCompartmentList():
                if aCompartment[1] == aCompartmentID:
                    if aCompartment[6] == '' or \
                       aCompartment[6] == 'default':
                        aPath = '/' + aCompartmentID
                        return aPath
                    else:
                        aPath = self.getPath( aCompartment[6] ) + '/' +\
                                aCompartmentID
                        return aPath
        elif self.Level == 2:
            for aCompartment in self.getCompartmentList():
                if( aCompartment[0] == aCompartmentID ):
                    if ( aCompartment[6] == '' or\
                         aCompartment[6] == 'default' ):
                        aPath = '/' + aCompartmentID
                        return aPath
                    else:
                        aPath = self.getPath( aCompartment[6] ) + '/' +\
                                aCompartmentID
                        return aPath

        else:
            raise Exception,"Version"+str(self.Level)+" ????"

    def getSpeciesReferenceID( self, aSpeciesID ):
        if self.Level == 1:
            for aSpecies in self.getSpeciesList():
                if aSpecies[1] == aSpeciesID:
                    return self.getPath( aSpecies[2] ) + ":" + aSpeciesID
                    
        elif self.Level == 2:
            for aSpecies in self.getSpeciesList():
                if aSpecies[0] == aSpeciesID:
                    return self.getPath( aSpecies[2] ) + ":" + aSpeciesID

        else:
            raise Exception,"Version"+str(self.Level)+" ????"

    def convertUnit( self, aValueUnit, aValue ):
        newValue = []
        if self.Level == 1:
            for unitList in self.getUnitDefinitionList():
                if unitList[1] == aValueUnit:
                    for anUnit in unitList[2]:
                        aValue = aValue * self.__getNewUnitValue( anUnit )
                newValue.append( aValue )
        elif self.Level == 2:
            for unitList in self.getUnitDefinitionList():
                if unitList[0] == aValueUnit:
                    for anUnit in unitList[2]:
                        aValue = aValue * self.__getNewUnitValue( anUnit )
                newValue.append( aValue )

        if newValue == []:
            return aValue
        else:
            return newValue[0]

    def __getNewUnitValue( self, anUnit ):
        aValue = 1

        # Scale
        if anUnit[2] != 0:
            aValue = aValue * pow( 10, anUnit[2] )

        # Multiplier
        aValue = aValue * anUnit[3]

        # Exponent
        aValue = pow( aValue, anUnit[1] )

        # Offset
        aValue = aValue + anUnit[4]

        return aValue

    def getSpeciesList( self ):
        " [[ Id , Name , Compartment , InitialAmount , InitialConcentration , SubstanceUnit , SpatialSizeUnit , Unit , HasOnlySubstanceUnit , BoundaryCondition , Charge , Constant ]] "
        if self.theSpeciesList != None:
            return self.theSpeciesList
        aList = []
        if self.theSBMLModel.getSpecies(0):
            for anIndex in range( self.theSBMLModel.getNumSpecies() ):
                aSpecies = self.theSBMLModel.getSpecies( anIndex )

                if aSpecies.isSetInitialAmount():
                    anInitialAmount_Sp = aSpecies.getInitialAmount()
                else:
                    anInitialAmount_Sp = "Unknown"
                if aSpecies.isSetInitialConcentration():
                    anInitialConcentration_Sp = aSpecies.getInitialConcentration()
                else:
                    anInitialConcentration_Sp = "Unknown"
                aListOfSpecies = [
                    aSpecies.getId(),
                    aSpecies.getName(),
                    aSpecies.getCompartment(),
                    anInitialAmount_Sp,
                    anInitialConcentration_Sp,
                    aSpecies.getSubstanceUnits(),
                    aSpecies.getSpatialSizeUnits(),
                    aSpecies.getUnits(),
                    aSpecies.getHasOnlySubstanceUnits(),
                    aSpecies.getBoundaryCondition(),
                    aSpecies.getCharge(),
                    aSpecies.getConstant()
                    ]
                aList.append( aListOfSpecies )
        self.theSpeciesList = aList
        return aList

    def getCompartmentList( self ):
        " [[ Id , Name , SpatialDimension , Size , Volume , Unit , Ouside , Constant ]] "
        if self.theCompartmentList != None:
            return self.theCompartmentList
        aList = []
        theList = self.theSBMLModel.getListOfCompartments()

        for anIndex in range( theList.getNumItems() ):
            if theList[anIndex].isSetSize():
                aSize = theList[anIndex].getSize()
            else:
                aSize = "Unknown"
            if theList[anIndex].isSetVolume():
                aVolume = theList[anIndex].getVolume()
            else:
                aVolume = "Unknown"

            aListOfCompartment = [
                theList[anIndex].getId(),
                theList[anIndex].getName(),
                theList[anIndex].getSpatialDimensions(),
                aSize,
                aVolume,
                theList[anIndex].getUnits(),
                theList[anIndex].getOutside(),
                theList[anIndex].getConstant()
                ]
            aList.append( aListOfCompartment )
        self.theCompartmentList = aList
        return aList

    def getEventList():
        " [[ Id , Name , StringTrigger , StringDelay , TimeUnit , [[ VariableAssignment , StringAssignment ]] ]] "
        if self.theEventList != None:
            return self.theEventList
        aList = []
        if self.theSBMLModel.getEvent(0):
            for anIndex in range( self.theSBMLModel.getNumEvents() ):
                anEvent = self.theSBMLModel.getEvent( anIndex )
                aListOfAListOfEvtAssignment = []
                if anEvent.getNumEventAssignments() > 0:
                    for anIndex_As in range( anEvent.getNumEventAssignments() ):
                        anEvtAssignment = anEvent.getEventAssignment( anIndex_As )
                        aListOfEvtAssignment = [
                            anEvtAssignment.getVariable(),
                            sub(
                                libsbml.formulaToString,
                                anEvtAssignment.getMath() )
                            ]
                        aListOfAListOfEvtAssignment.append(
                            aListOfEvtAssignment )
                aListOfEvent = [
                    anEvent.getId(),
                    anEvent.getName(),
                    sub( libsbml.formulaToString, anEvent.getTrigger() ),
                    sub( libsbml.formulaToString, anEvent.getDelay() ),
                    anEvent.getTimeUnits(),
                    aListOfAListOfEvtAssignment
                    ]
                aList.append( aListOfEvent )
        return aList

    def getFunctionDefinitionMap( self ):
        " [[ Id , Name , String ]] "
        if self.theFunctionDefinitionMap != None:
            return self.theFunctionDefinitionMap
        aFuncDefMap = {}
        for anIndex in range( aSBMLmodel.getNumFunctionDefinitions() ):
            aFuncDef = aSBMLmodel.getFunctionDefinition( anIndex )
            aFuncDefMap[ FunctionDefinition.getId() ] = [
                aFuncDef.getName(),
                sub( libsbml.formulaToString, aFuncDef.getMath() )
                ]
        self.theFunctionDefinitionMap = aFuncDefMap
        return aFuncDefMap

    def getParameterList( self ):
        " [[ Id , Name , Value , Unit , Constant ]] "
        if self.theParameterList != None:
            return self.theParameterList
        aList = []
        if self.theSBMLModel.getParameter(0):
            NumParameter = self.theSBMLModel.getNumParameters()
            for anIndex in range( NumParameter ):
                aParameter = self.theSBMLModel.getParameter( anIndex )

                if aParameter.isSetValue():
                    aValue_Pa = aParameter.getValue()
                else:
                    aValue_Pa = 'Unknown'
                aListOfParameter = [
                    aParameter.getId(),
                    aParameter.getName(),
                    aValue_Pa,
                    aParameter.getUnits(),
                    aParameter.getConstant()
                    ]

                aList.append( aListOfParameter )
        self.theParameterList = aList
        return aList

    def getReactionList( self ):
        " [[ Id , Name , [ Formula , String , TimeUnit , SubstanceUnit , [[ ParameterId , ParameterName , ParameterValue , ParameterUnit , ParameterConstant ]] ] , Reversible , Fast , [[ ReactantSpecies , ( ReactantStoichiometry , ReactantStoichiometryMath ) , ReactantDenominator  ]] , [[  ProductSpecies , ( ProductStoichiometry , ProductStoichiometryMath ) , ProductDenominator ]] , [[ ModifierSpecies ]] ]] "
        if self.theReactionList != None:
            return self.theReactionList
        aList = []
        if self.theSBMLModel.getReaction(0):
            for anIndex in range( self.theSBMLModel.getNumReactions() ):
                aReaction = self.theSBMLModel.getReaction( anIndex )

                if aReaction.isSetKineticLaw():
                    aKineticLaw = aReaction.getKineticLaw()
                    if aKineticLaw != []:
                        if aKineticLaw.isSetFormula():
                            aFormula_KL = aKineticLaw.getFormula()
                        else:
                            aFormula_KL = ''
                        aMath = []
                        if self.Level == 1:
                            aMath.append( '' )
                        else:
                            if aKineticLaw.isSetMath():
                                anASTNode_KL = aKineticLaw.getMath()
                                aMath.append(
                                    libsbml.formulaToString(
                                        anASTNode_KL ) )
                            else:
                                aMath.append( '' )
                        aListOfAListOfParameter = []
                        for anIndexPara in range( aKineticLaw.getNumParameters() ):
                            aParameter = aKineticLaw.getParameter( anIndexPara )

                            aListOfParameter = [
                                aParameter.getId(),
                                aParameter.getName(),
                                str( aParameter.getValue() ),
                                aParameter.getUnits(),
                                aParameter.getConstant()
                                ]
                            aListOfAListOfParameter.append( aListOfParameter )
                        anExpressionAnnotation = aKineticLaw.getAnnotation()
                        aListOfKineticLaw = [
                            aFormula_KL,
                            aMath,
                            aKineticLaw.getTimeUnits(),
                            aKineticLaw.getSubstanceUnits(),
                            aListOfAListOfParameter,
                            anExpressionAnnotation
                            ]
                else:
                    aListOfKineticLaw = None

                aListOfAListOfReactant = []
                for anIndexR in range( aReaction.getNumReactants() ):
                    aSpeciesReference = aReaction.getReactant( anIndexR )
                    aStoichiometry_R = aSpeciesReference.getStoichiometry()

                    if aSpeciesReference.isSetStoichiometryMath():
                        anASTNode_R = aSpeciesReference.getStoichiometryMath()
                        aString_R = sub( libsbml.formulaToString , anASTNode_R )
                    else:
                        aString_R = []

                    aDenominator_R = aSpeciesReference.getDenominator()

                    aListOfReactant = []
                    aListOfReactant.append( aSpeciesReference.getSpecies() )
                    # XXX: this sucks.
                    # [
                    #   Species,
                    #   ( Stoichiometry, )
                    #   ( StoichiometryMath, )
                    #   Denominator
                    # ]
                    if aString_R == []:
                        aListOfReactant.append( aStoichiometry_R )
                    else:
                        if aStoichiometry_R == []:
                            aListOfReactant.append( aString_R )
                        aListOfReactant.append( aString_R )
                    aListOfReactant.append( aDenominator_R )
                    aListOfAListOfReactant.append( aListOfReactant )

                aListOfAListOfProduct = []
                for anIndexP in range( aReaction.getNumProducts() ):
                    aSpeciesReference = aReaction.getProduct( anIndexP )
                    aStoichiometry_P = aSpeciesReference.getStoichiometry()

                    if aSpeciesReference.isSetStoichiometryMath():
                        anASTNode_P = aSpeciesReference.getStoichiometryMath()
                        aString_P = sub( libsbml.formulaToString , anASTNode_P )
                    else:
                        aString_P = []

                    aDenominator_P = aSpeciesReference.getDenominator()

                    aListOfProduct = []
                    aListOfProduct.append( aSpeciesReference.getSpecies() )
                    # XXX: this sucks.
                    # [
                    #   Species,
                    #   ( Stoichiometry, )
                    #   ( StoichiometryMath, )
                    #   Denominator
                    # ]
                    if aString_P == []:
                        aListOfProduct.append( aStoichiometry_P )
                    else:
                        if aStoichiometry_P != []:
                            aListOfProduct.append( aStoichiometry_P )
                        aListOfProduct.append( aString_P )
                    aListOfProduct.append( aDenominator_P )
                    aListOfAListOfProduct.append( aListOfProduct )


                aListOfModifier = []
                for anIndexM in range( aReaction.getNumModifiers() ):
                    aSpeciesReference = aReaction.getModifier( anIndexM )
                    aListOfModifier.append( aSpeciesReference.getSpecies() )

                aListOfReaction = [
                    aReaction.getId(),
                    aReaction.getName(),
                    aListOfKineticLaw,
                    aReaction.getReversible(),
                    aReaction.getFast(),
                    aListOfAListOfReactant,
                    aListOfAListOfProduct,
                    aListOfModifier
                    ]
                aList.append( aListOfReaction )
        self.theReactionList = aList
        return aList

    def getUnitDefinitionList( self ):
        " [[ Id , Name , [[ Kind , Exponent , Scale , Multiplier , Offset ]] ]] "
        if self.theUnitDefinitionList != None:
            return self.theUnitDefinitionList
        aList = []
        if self.theSBMLModel.getUnitDefinition(0):
            for anIndex in range( self.theSBMLModel.getNumUnitDefinitions() ):
                aListOfUnitDefinition = []
                anUnitDefinition = self.theSBMLModel.getUnitDefinition( anIndex )
                aListOfAListOfUnit = []
                if anUnitDefinition.getUnit(0):
                    NumUnit = anUnitDefinition.getNumUnits()
                    for anIndexU in range( NumUnit ):
                        anUnit = anUnitDefinition.getUnit( anIndexU )
                        ListOfUnit = [
                            libsbml.UnitKind_toString( anUnit.getKind() ),
                            anUnit.getExponent(),
                            anUnit.getScale(),
                            anUnit.getMultiplier(),
                            anUnit.getOffset()
                            ]
                        aListOfAListOfUnit.append( ListOfUnit )
                aListOfUnitDefinition = [
                    anUnitDefinition.getId(),
                    anUnitDefinition.getName(),
                    aListOfAListOfUnit
                    ]
                aList.append( aListOfUnitDefinition )
        self.theUnitDefinitionList = aList
        return aList

    def getRuleList( self ):
        " [[ RuleType, Formula, Variable ]] "
        if self.theRuleList != None:
            return self.theRuleList
        aList = []
        if self.theSBMLModel.getRule(0):
            NumRule = self.theSBMLModel.getNumRules()
            for Num in range( NumRule ):
                aRule = self.theSBMLModel.getRule( Num )

                if aRuleType == libsbml.SBML_ALGEBRAIC_RULE:
                    aVariable = ''
                elif aRuleType == libsbml.SBML_ASSIGNMENT_RULE or \
                     aRuleType == libsbml.SBML_RATE_RULE:
                    aVariable = aRule.getVariable()
                elif aRuleType == libsbml.SBML_SPECIES_CONCENTRATION_RULE:
                    aVariable = aRule.getSpecies()
                elif aRuleType == libsbml.SBML_COMPARTMENT_VOLUME_RULE:
                    aVariable = aRule.getCompartment()
                elif aRuleType == libsbml.SBML_PARAMETER_RULE:
                    aVariable = aRule.getName()
                else:
                    raise TypeError, " The type of Rule must be Algebraic, Assignment or Rate Rule"
                aListOfRule = [
                    aRule.getTypeCode(),
                    aRule.getFormula(),
                    aVariable
                    ]

                aList.append( aListOfRule )
        self.theRuleList = aList
        return aList

class Compartment( Model ):
    def __init__( self, aModel ):
        self.theModel = aModel

    def initialize( self, aCompartment ):
        self.__setSizeToDictionary( aCompartment )
        self.__setUnitToDictionary( aCompartment )

    def getCompartmentID( self, aCompartment ):
        if aCompartment[6] == '':
            if self.theModel.Level == 1:
                aSystemID = '/:' + aCompartment[1]
            elif self.theModel.Level == 2:
                aSystemID = '/:' + aCompartment[0]
            else:
                raise NameError,"Compartment Class needs a ['ID']"

        else:
            if( self.theModel.Level == 1 ):
                aSystemID = self.theModel.getPath( aCompartment[6] ) + ':'+ aCompartment[1]
            elif( self.theModel.Level == 2 ):
                aSystemID = self.theModel.getPath( aCompartment[6] ) + ':'+ aCompartment[0]
        return 'System:' + aSystemID

    def __setSizeToDictionary( self, aCompartment ):

        if( self.theModel.Level == 1 ):
            if( aCompartment[4] != "Unknown" ):
                self.theModel.CompartmentSize[ aCompartment[1] ] = aCompartment[4]

            else:
                self.theModel.CompartmentSize[ aCompartment[1] ] = self.__getOutsideSize( aCompartment[6] )
                
        elif( self.theModel.Level == 2 ):
            if( aCompartment[3] != "Unknown" ):
                self.theModel.CompartmentSize[ aCompartment[0] ] = aCompartment[3]

            else:
                self.theModel.CompartmentSize[ aCompartment[0] ] = self.__getOutsideSize( aCompartment[6] )

    def __setUnitToDictionary( self, aCompartment ):

        if( self.theModel.Level == 1 ):
            aCompartmentID = aCompartment[1]

        elif( self.theModel.Level == 2 ):
            aCompartmentID = aCompartment[0]


        if( aCompartment[5] != '' ):
            self.theModel.CompartmentUnit[ aCompartmentID ] = aCompartment[5]

        else:
            self.theModel.CompartmentUnit[ aCompartmentID ] = self.__getOutsideUnit( aCompartment[6] )

    def __getOutsideSize( self, anOutsideCompartment ):
        if anOutsideCompartment == '':
            return float( 1 )
        else:
            return self.theModel.CompartmentSize[ anOutsideCompartment ]

    def __getOutsideUnit( self, anOutsideCompartment ):
        if anOutsideCompartment == '':
            return ''
        else:
            return self.theModel.CompartmentUnit[ anOutsideCompartment ]

    def getCompartmentSize( self, aCompartment ):
        if self.theModel.Level == 1:
            return self.theModel.CompartmentSize[ aCompartment[1] ]

        elif self.theModel.Level == 2:

            return self.theModel.CompartmentSize[ aCompartment[0] ]

    def getCompartmentUnit( self, aCompartment ):
        if self.theModel.Level == 1:
            return self.theModel.CompartmentUnit[ aCompartment[1] ]
        elif self.theModel.Level == 2:
            return self.theModel.CompartmentUnit[ aCompartment[0] ]

class Species( Model ):
    def __init__( self, aModel ):
        self.theModel = aModel
    
    def getSpeciesID( self, aSpecies ):
        aCompartmentID = aSpecies[2]

        if aCompartmentID == '':
            raise NameError, 'compartment property of Species must be defined'

        if self.theModel.Level == 1:
            aSystemID = self.theModel.getPath( aCompartmentID ) + ':' + aSpecies[1]

        elif self.theModel.Level == 2:
            aSystemID = self.theModel.getPath( aCompartmentID ) + ':' + aSpecies[0]
        else:
            raise Exception,"Version"+str(self.Level)+" ????"
                
        return 'Variable:' + aSystemID

    def getSpeciesValue( self, aSpecies ):
        if aSpecies[ 3 ] != 'Unknown': # initialAmount
            return float( aSpecies[ 3 ] )
        elif self.theModel.Level == 2 and \
             aSpecies[ 4 ] != 'Unknown': # initialConcentration
            # spatialSizeUnits and hasOnlySubstanceUnits should be checked
            aSize = self.theModel.CompartmentSize[ aSpecies[ 2 ] ]
            return aSpecies[ 4 ] * aSize
        else:
            raise ValueError, 'InitialAmount or InitialConcentration of Species [%s] must be defined.' % ( aSpecies[ 0 ] )

    def getConstant( self, aSpecies ):
        if self.theModel.Level == 1:
            if aSpecies[9] == 1:
                return 1
            else:
                return 0
        elif self.theModel.Level == 2:
            if aSpecies[11] == 1:
                return 1
            else:
                return 0

class Rule( Model ):
    def __init__( self, aModel ):

        self.theModel = aModel
        self.RuleNumber = 0

    def initialize( self ):

        self.VariableReferenceList = []
        self.VariableNumber = 0
        self.ParameterNumber = 0
        self.RuleNumber = self.RuleNumber + 1

    def getRuleID( self ):

        return 'Process:/SBMLRule:Rule' + str( self.RuleNumber )

    def getVariableType( self, aName ):
        for aSpecies in self.theModel.getSpeciesList():
            if ( self.theModel.Level == 1 and aSpecies[1] == aName ) or \
               ( self.theModel.Level == 2 and aSpecies[0] == aName ):
                return libsbml.SBML_SPECIES

        for aParameter in self.theModel.getParameterList():
            if ( self.theModel.Level == 1 and aParameter[1] == aName ) or \
               ( self.theModel.Level == 2 and aParameter[0] == aName ):
                return libsbml.SBML_PARAMETER

        for aCompartment in self.theModel.getCompartmentList():
            if ( self.theModel.Level == 1 and aCompartment[1] == aName ) or \
               ( self.theModel.Level == 2 and aCompartment[0] == aName ):
                return libsbml.SBML_COMPARTMENT

        raise TypeError, "Variable type must be Species, Parameter, or Compartment"
    
    def setSpeciesToVariableReference( self, aName, aStoichiometry='0' ):

        for aSpecies in self.theModel.getSpeciesList():
            if ( self.theModel.Level == 1 and aSpecies[1] == aName ) or \
               ( self.theModel.Level == 2 and aSpecies[0] == aName ):
                for c in range( len( self.VariableReferenceList ) ):
                    aVariableReference = self.VariableReferenceList[ c ]
                    if aVariableReference[1].split(':')[2] == aName:
                        if aStoichiometry != 0:
                            aVariableReference[ 2 ] = aStoichiometry
                        compartmentName = self.setCompartmentToVariableReference( aSpecies[ 2 ] )
                        return ( aVariableReference[ 0 ], compartmentName )

                aVariableList = []

                variableName = 'V%d' % ( self.VariableNumber )
                aVariableList.append( variableName )
                self.VariableNumber = self.VariableNumber + 1

                aVariableID = self.theModel.getSpeciesReferenceID( aName )
                aVariableList.append( 'Variable:' + aVariableID )
                aVariableList.append( aStoichiometry )
                
                self.VariableReferenceList.append( aVariableList )
                
                compartmentID = aSpecies[ 2 ]
                compartmentName = self.setCompartmentToVariableReference( compartmentID )

                return ( variableName, compartmentName )

    def setParameterToVariableReference( self, aName, aStoichiometry='0' ):
        for aParameter in self.theModel.getParameterList():
            if ( self.theModel.Level == 1 and aParameter[1] == aName ) or \
               ( self.theModel.Level == 2 and aParameter[0] == aName ):
                for c in range( len( self.VariableReferenceList ) ):
                    aVariableReference = self.VariableReferenceList[ c ]
                    if aVariableReference[1].split(':')[2] == aName:
                        if aStoichiometry != 0:
                            aVariableReference[ 2 ] = aStoichiometry
                        return aVariableReference[ 0 ]
                aParameterList = []
                variableName = 'P%d' % ( self.ParameterNumber )
                aParameterList.append( variableName )
                self.ParameterNumber = self.ParameterNumber + 1
                aParameterList.append( 'Variable:/SBMLParameter:' + aName )
                aParameterList.append( aStoichiometry )
                self.VariableReferenceList.append( aParameterList )
                return variableName
        return None

    def setCompartmentToVariableReference( self, aName, aStoichiometry='0' ):
        for aCompartment in self.theModel.getCompartmentList():
            if ( self.theModel.Level == 1 and aCompartment[1] == aName ) or \
               ( self.theModel.Level == 2 and aCompartment[0] == aName ):
                for c in range( len( self.VariableReferenceList ) ):
                    aVariableReference = self.VariableReferenceList[ c ]
                    if aVariableReference[1].split(':')[1] == \
                         self.theModel.getPath( aName ) and \
                       aVariableReference[1].split(':')[2] == 'SIZE':
                        if aStoichiometry != 0:
                            aVariableReference[ 2 ] = aStoichiometry
                        return aVariableReference[ 0 ]
                aCompartmentList = []
                aCompartmentList.append( aName )
                aCompartmentList.append(
                    'Variable:' + self.theModel.getPath( aName ) + ':SIZE' )
                aCompartmentList.append( aStoichiometry )
                self.VariableReferenceList.append( aCompartmentList )
                return aName

    def __convertVariableName( self, anASTNode ):
        aNumChildren = anASTNode.getNumChildren()
        if aNumChildren == 2:
            self.__convertVariableName( anASTNode.getLeftChild() )
            self.__convertVariableName( anASTNode.getRightChild() )
        elif aNumChildren == 1:
            self.__convertVariableName( anASTNode.getLeftChild() )
        elif aNumChildren == 0:
            if anASTNode.isNumber() == 1:
                pass
            else:
                aName = anASTNode.getName()
                aType = self.getVariableType( aName )

                # Species
                if ( aType == libsbml.SBML_SPECIES ):
                    ( variableName, compartmentName ) = self.setSpeciesToVariableReference( aName )
                    if( variableName != '' ):

                        anASTNode.setType( libsbml.AST_DIVIDE )
                        anASTNode.addChild( libsbml.ASTNode( libsbml.AST_NAME ) )
                        anASTNode.addChild( libsbml.ASTNode( libsbml.AST_NAME ) )
                        anASTNode.getLeftChild().setName( '%s.Value' % ( variableName ) )      
                        anASTNode.getRightChild().setName( '%s.Value' % ( compartmentName ) )      
                        return anASTNode

                # Parameter
                elif aType == libsbml.SBML_PARAMETER:
                    
                    variableName = self.setParameterToVariableReference( aName )
                    if( variableName != '' ):
                        anASTNode.setName( '%s.Value' % ( variableName ) )
                        return anASTNode

                # Compartment
                elif aType == libsbml.SBML_COMPARTMENT:
                    
                    variableName = self.setCompartmentToVariableReference( aName )
                    if( variableName != '' ):
                        anASTNode.setName( '%s.Value' % ( variableName ) )
                        return anASTNode

        return anASTNode

    def convertRuleFormula( self, aFormula ):

        aASTRootNode = libsbml.parseFormula( aFormula )

        convertedAST = self.__convertVariableName( aASTRootNode )
        convertedFormula = libsbml.formulaToString( convertedAST )
        
        return convertedFormula

class Reaction( Model ):
    def __init__( self, aModel ):
        self.theModel = aModel

    def initialize( self ):

        self.SubstrateNumber = 0
        self.ProductNumber = 0
        self.ModifierNumber = 0
        self.ParameterNumber = 0

        self.VariableReferenceList = []

    def getReactionID( self, aReaction ):

        if self.theModel.Level == 1:
            if aReaction[1] != '':
                return 'Process:/:' + aReaction[1]
            else:
                raise NameError,"Reaction must set the Reaction name"
                
        elif self.theModel.Level == 2:
            if aReaction[0] != '':
                return 'Process:/:' + aReaction[0]
            else:
                raise NameError,"Reaction must set the Reaction ID"

    def setCompartmentToVariableReference( self, aName ):

        for aCompartment in self.theModel.getCompartmentList():
            if aCompartment[0] == aName or \
               aCompartment[1] == aName:

                for aVariableReference in self.VariableReferenceList:
                    if aVariableReference[1].split(':')[2] == 'SIZE':
                        aCurrentPath = ( aVariableReference[1].split(':')[1] )
                        aLastSlash = string.rindex( aCurrentPath, '/' )
                        variableName = aCurrentPath[aLastSlash+1:]
                        return aVariableReference[ 0 ]
                        ## return variableName
                                
                aCompartmentList = []
                aCompartmentList.append( aName )
                            
                aCompartmentList.append(
                    'Variable:' + self.theModel.getPath( aName ) + ':SIZE' )
                            
                aCompartmentList.append( '0' )
                self.VariableReferenceList.append( aCompartmentList )

                return aCompartmentList[0]

        return ''

    def __convertVariableName( self, anASTNode ):
        aNumChildren = anASTNode.getNumChildren()
        if aNumChildren == 2:
            self.__convertVariableName( anASTNode.getLeftChild() )
            self.__convertVariableName( anASTNode.getRightChild() )
            return anASTNode
        elif aNumChildren == 1:
            self.__convertVariableName( anASTNode.getLeftChild() )
            return anASTNode
        elif aNumChildren == 0:
            if anASTNode.isNumber() == 1:
                pass
            else:
                aName = anASTNode.getName()
                variableName = ''
                
                for aSpecies in self.theModel.getSpeciesList():
                    if aSpecies[0] == aName or aSpecies[1] == aName:
                        for aVariableReference in self.VariableReferenceList:
                            if aVariableReference[1].split(':')[2] == aName:
                                variableName =  aVariableReference[0]
                        if( self.theModel.Level == 2 and variableName == '' ):
                            raise NameError,"in libSBML :"+aName+" isn't defined in VariableReferenceList"
                        elif( self.theModel.Level == 1 and variableName == '' ):
                            aModifierList = []
                            aModifierList.append(
                                'C' + str( self.ModifierNumber ) )
                            self.ModifierNumber = self.ModifierNumber + 1
                            
                            aModifierID = self.theModel.getSpeciesReferenceID( aName )
                            aModifierList.append( 'Variable:' + aModifierID )
                            aModifierList.append( '0' )
                            self.VariableReferenceList.append( aModifierList )
                            variableName = aModifierList[0]
                        compartmentName = self.setCompartmentToVariableReference( aSpecies[ 2 ] )
                        anASTNode.setType( libsbml.AST_DIVIDE )
                        anASTNode.addChild( libsbml.ASTNode( libsbml.AST_NAME ) )
                        anASTNode.addChild( libsbml.ASTNode( libsbml.AST_NAME ) )
                        anASTNode.getLeftChild().setName( '%s.Value' % ( variableName ) )      
                        anASTNode.getRightChild().setName( '%s.Value' % ( compartmentName ) )      
                        return anASTNode
                for aParameter in self.theModel.getParameterList():
                    if aParameter[0] == aName or \
                       aParameter[1] == aName:
                        for aVariableReference in self.VariableReferenceList:
                            if aVariableReference[1].split(':')[2] == aName:
                                variableName = aVariableReference[0]
                        if( variableName == '' ):
                            aParameterList = []
                            aParameterList.append(
                                'Param' + str( self.ParameterNumber ) )
                            self.ParameterNumber = self.ParameterNumber + 1
                            aParameterList.append(
                                'Variable:/SBMLParameter:' + aName )
                            aParameterList.append( '0' )
                            self.VariableReferenceList.append( aParameterList )
                            variableName = aParameterList[0]
                        anASTNode.setName( '%s.Value' % ( variableName ) )
                        return anASTNode
                variableName = self.setCompartmentToVariableReference( aName )
                if variableName != '':
                    anASTNode.setName( '%s.Value' % ( variableName ) )
                    return anASTNode
                return anASTNode

    def convertKineticLawFormula( self, aFormula ):

        aASTRootNode = libsbml.parseFormula( aFormula )
        convertedAST = self.__convertVariableName( aASTRootNode )

        return libsbml.formulaToString( convertedAST )

    def getStoichiometry( self, aSpeciesID, aStoichiometry ):
        if self.theModel.Level == 1:
            for aSpecies in self.theModel.getSpeciesList():
                if aSpecies[1] == aSpeciesID:
                    if aSpecies[9] == 1:
                        return int( 0 )
                    else:
                        return int( aStoichiometry )
        elif self.theModel.Level == 2:
            for aSpecies in self.theModel.getSpeciesList():
                if aSpecies[0] == aSpeciesID:
                    if aSpecies[11] == 1:
                        return int( 0 )
                    else:
                        return int( aStoichiometry )
        else:
           raise Exception, "Version" + str(self.Level) + " ????"

class Parameter( Model ):
   def __init__( self, aModel ):
        self.theModel = aModel

   def getParameterID( self, aParameter ):
       if self.theModel.Level == 1:
           if aParameter[1] != '':
               return 'Variable:/SBMLParameter:' + aParameter[1]
           else:
               raise NameError, "Parameter must set the Parameter Name"
       elif self.theModel.Level == 2:
           if aParameter[0] != '':
               return 'Variable:/SBMLParameter:' + aParameter[0]
           else:
               raise NameError, "Parameter must set the Parameter ID"
       else:
           raise Exception, "Version" + str(self.Level) + " ????"
 
   def getParameterValue( self, aParameter ):
       return aParameter[ 2 ]

class Event( Model ):
    def __init__( self, aModel ):
        self.theModel = aModel
        self.EventNumber = 0

    def getEventID( self, aEvent ):
        if aEvent[0] != '':
            return 'Process:/:' + aEvent[0]
        elif aEvent[1] != '':
            return 'Process:/:' + aEvent[1]
        else:
            anID = 'Process:/:Event' + self.EventNumber
            self.EventNumber = self.EventNumber + 1
            return anID

def convert( aSBMLString ):
    aSBMLDocument = libsbml.readSBMLFromString( aSBMLString )

    theModel = Model( aSBMLDocument )
    theCompartment = Compartment( theModel )
    theParameter = Parameter( theModel )
    theSpecies = Species( theModel )
    theRule = Rule( theModel )
    theReaction = Reaction( theModel )

    anEml = Eml()

    # -- Set Stepper
    anEml.createStepper( 'ODEStepper', 'DE' )

    # -- Set Compartment ( System )
    # setFullID
    aSystemFullID='System::/'
    anEml.createEntity( 'System', aSystemFullID )
    anEml.setEntityProperty( aSystemFullID, 'StepperID', ['DE'] )
    anEml.setEntityProperty( aSystemFullID, 'Name', ['Default'] )

    for aCompartment in theModel.getCompartmentList():
        # initialize
        theCompartment.initialize( aCompartment )

        # getPath
        if theModel.Level == 1:
            aPath = theModel.getPath( aCompartment[1] )
        elif theModel.Level == 2:
            aPath = theModel.getPath( aCompartment[0] )           
        
        # setFullID
        if aPath == '/':
            aSystemFullID = 'System::/'
        else:
            aSystemFullID = theCompartment.getCompartmentID( aCompartment )
            anEml.createEntity( 'System', aSystemFullID )
            # setStepper 
            anEml.setEntityProperty( aSystemFullID, 'StepperID', ['DE'] )
        # setName( default = [] )
        if theModel.Level == 2:
            if aCompartment[1] != '':
                anEml.setEntityProperty(
                    aSystemFullID, 'Name', aCompartment[1:2] )
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
        if aSizeUnit != '':
            aSizeValue = theModel.convertUnit( aSizeUnit, aSizeValue )

        aTmpList = [ str( aSizeValue ) ]
        anEml.setEntityProperty( aSizeFullID, 'Value', aTmpList[0:1] )


        # setConstant( default = 1 )
        if aCompartment[7] == 1:
            anEml.setEntityProperty( aSizeFullID, 'Fixed', ['1',] )
           
    # -- Set GlobalParameter ( Variable )
    if len( theModel.getParameterList() ) > 0:
        # setGlobalParameterSystem
        aSystemFullID='System:/:SBMLParameter'
        anEml.createEntity( 'System', aSystemFullID )
        anEml.setEntityProperty( aSystemFullID, 'StepperID', ['DE'] )
        anEml.setEntityProperty( aSystemFullID, 'Name', ['Global Parameter'] )

    for aParameter in theModel.getParameterList():
        # setFullID
        aSystemFullID = theParameter.getParameterID( aParameter )
        anEml.createEntity( 'Variable', aSystemFullID )
        # setName
        if aParameter[1] != '':
            anEml.setEntityProperty( aSystemFullID, 'Name', aParameter[1:2] )
        # setValue
        aTmpList = [ str( theParameter.getParameterValue( aParameter ) ) ]
        anEml.setEntityProperty( aSystemFullID, 'Value', aTmpList[0:1] )
        # setFixed ( default = 1 )
        if aParameter[4] == 1:
            # aTmpList = [ str( aParameter[4] ) ]
            aTmpList = [ '1' ]
            anEml.setEntityProperty( aSystemFullID, 'Fixed', aTmpList[0:1] )

    # -- Set Species ( Variable )
    # set FullID
    for aSpecies in theModel.getSpeciesList():
        aSystemFullID = theSpecies.getSpeciesID( aSpecies )
        anEml.createEntity( 'Variable', aSystemFullID )
        # set name
        if theModel.Level == 2:
            if aSpecies[1] != '':
                anEml.setEntityProperty( aSystemFullID, 'Name', aSpecies[1:2] )
        # set value
        aTmpList = [ str( theSpecies.getSpeciesValue( aSpecies ) ) ]
        anEml.setEntityProperty( aSystemFullID, 'Value', aTmpList[0:1] )
        # setFixed
        aConstant = theSpecies.getConstant( aSpecies )
        anEml.setEntityProperty(
            aSystemFullID, 'Fixed', [ str( aConstant ) ] )

    # -- Set Rule ( Process )
    if len( theModel.getRuleList() ) > 0:
        # make Rule System #
        aSystemFullID='System:/:SBMLRule'
        anEml.createEntity( 'System', aSystemFullID )
        anEml.setEntityProperty( aSystemFullID,
                                 'Name',
                                 ['System for SBML Rule'] )
        anEml.setEntityProperty( aSystemFullID, 'StepperID', ['DE'] )

    for aRule in theModel.getRuleList():
        theRule.initialize()
        # set FullID
        aSystemFullID = theRule.getRuleID()
        # Algebraic Rule
        if aRule[0] == libsbml.SBML_ALGEBRAIC_RULE:
            anEml.createEntity( 'ExpressionAlgebraicProcess', aSystemFullID )
        # Assignment Rule #
        elif aRule[0] == libsbml.SBML_ASSIGNMENT_RULE or \
             aRule[0] == libsbml.SBML_SPECIES_CONCENTRATION_RULE or \
             aRule[0] == libsbml.SBML_COMPARTMENT_VOLUME_RULE or \
             aRule[0] == libsbml.SBML_PARAMETER_RULE:
            anEml.createEntity( 'ExpressionAssignmentProcess', aSystemFullID )
            aVariableType = theRule.getVariableType( aRule[2] )
            if aVariableType == libsbml.SBML_SPECIES:
                theRule.setSpeciesToVariableReference( aRule[2], '1' )
            elif aVariableType == libsbml.SBML_PARAMETER:
                theRule.setParameterToVariableReference( aRule[2], '1' )
            elif aVariableType == libsbml.SBML_COMPARTMENT:
                theRule.setCompartmentToVariableReference( aRule[2], '1' )
            else:
                raise TypeError, \
                    "Variable type must be Species, Parameter, or Compartment"
        # Rate Rule #
        elif aRule[0] == libsbml.SBML_RATE_RULE:
            anEml.createEntity( 'ExpressionFluxProcess', aSystemFullID )
            aVariableType = theRule.getVariableType( aRule[2] )
            if aVariableType == libsbml.SBML_SPECIES:
                theRule.setSpeciesToVariableReference( aRule[2], '1' )
            elif aVariableType == libsbml.SBML_PARAMETER:
                theRule.setParameterToVariableReference( aRule[2], '1' )
            elif aVariableType == libsbml.SBML_COMPARTMENT:
                theRule.setCompartmentToVariableReference( aRule[2], '1' )
            else:
                raise TypeError, \
                    "Variable type must be Species, Parameter, or Compartment"
        else:
            raise TypeError, \
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
    # -- Set Reaction ( Process )
    for aReaction in theModel.getReactionList():
        theReaction.initialize()
        # set FullID
        aSystemFullID = theReaction.getReactionID( aReaction )
        anEml.createEntity( 'ExpressionFluxProcess', aSystemFullID )
        # setName
        if theModel.Level == 2:
            if aReaction[1] != '':
                anEml.setEntityProperty( aSystemFullID, 'Name', aReaction[1:2] )
        # setSubstrate
        for aSubstrate in aReaction[5]:
            aSubstrateList = []
            aSubstrateList.append( 'S' + str( theReaction.SubstrateNumber ) )
            theReaction.SubstrateNumber = theReaction.SubstrateNumber + 1
            aSubstrateID = theModel.getSpeciesReferenceID( aSubstrate[0] )
            if aSubstrateID == None:
                raise NameError,"Species "+aSubstrate[0]+" not found"
            aSubstrateList.append( 'Variable:' + aSubstrateID )
            if aSubstrate[2] != 1:
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
            if aProductID == None:
                raise NameError,"Species "+aProduct[0]+" not found"
            aProductList.append( 'Variable:' + aProductID )
            if aProduct[2] != 1:
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
            if aModifierID == None:
                raise NameError, "Species "+aModifier[0]+" not found"
            aModifierList.append( 'Variable:' + aModifierID )
            aModifierList.append( '0' )
            theReaction.VariableReferenceList.append( aModifierList )


        # setProperty
        if aReaction[2] != []:
            if aReaction[2][4] != []:
                for aParameter in aReaction[2][4]:
                    if aParameter[2] != '': 
                        aTmpList = [ str( aParameter[2] ) ]
                        if theModel.Level == 1:
                            anEml.setEntityProperty(
                                aSystemFullID, aParameter[1], aTmpList[0:1] )
                        elif theModel.Level == 2:
                            anEml.setEntityProperty(
                                aSystemFullID, aParameter[0], aTmpList[0:1] )
                            
                          
            # -- set "Expression" Property
            # convert SBML format formula to E-Cell format formula
            if( aReaction[2][0] != '' ):
                anExpression = [
                    str( theReaction.convertKineticLawFormula( aReaction[2][0] ) ) ]
                # set Expression Property for ExpressionFluxProcess
                anEml.setEntityProperty( aSystemFullID,
                                         'Expression',
                                         anExpression )
                # setVariableReferenceList
                anEml.setEntityProperty( aSystemFullID,
                                         'VariableReferenceList',
                                         theReaction.VariableReferenceList )
    return anEml.asString()
