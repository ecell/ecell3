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
import libsbml
import math, sys, re
import decimal, fractions

#---Other functions are using this 'sub'funtion in this file.---
def sub( fun , indata ):
    if indata != []:
        if fun( indata ):
            outdata = fun( indata )
        else:
            outdata = []
    else:
        outdata = []

    return outdata
#---------------------------------------------------------------


def get_Compartment_list( aSBMLmodel ):
    " [[ Id , Name , SpatialDimension , Size , Volume , Unit , Ouside , Constant ]] "
    LIST = []
    theList = aSBMLmodel.getListOfCompartments()
    #if Model_getCompartment( aSBMLmodel , 0 ):
        
    NumCompartment = len( theList )
        
    for Num in range( NumCompartment ):
        aCompartmentDic = {}
            #aCompartment = Model_getCompartment( aSBMLmodel , Num )

        anId = theList[Num].getId()
            #anId = sub( Compartment_getId , aCompartment )
        aName = theList[Num].getName()
            #aName = sub( Compartment_getName , aCompartment )
        aSpatialDimension = theList[Num].getSpatialDimensions()
            #aSpatialDimension = Compartment_getSpatialDimensions( aCompartment )
        if theList[Num].isSetSize():
            aSize = theList[Num].getSize()
                #aSize = sub( Compartment_getSize , aCompartment )
        else:
            aSize = "Unknown"

        if theList[Num].isSetVolume():
            aVolume = theList[Num].getVolume()
                #aVolume = sub( Compartment_getVolume , aCompartment )
        else:
            aVolume = "Unknown"

        anUnit = theList[Num].getUnits()
            #anUnit = sub( Compartment_getUnits , aCompartment )
        anOutside = theList[Num].getOutside()
            #anOutside = sub( Compartment_getOutside , aCompartment )
        aConstant = theList[Num].getConstant()
            #aConstant = Compartment_getConstant( aCompartment )

        aCompartmentDic[ 'Id' ]               =  anId 
        aCompartmentDic[ 'Name' ]             =  aName 
        aCompartmentDic[ 'SpatialDimension' ] =  aSpatialDimension 
        aCompartmentDic[ 'Size' ]             =  aSize 
        aCompartmentDic[ 'Volume' ]           =  aVolume 
        aCompartmentDic[ 'Unit' ]             =  anUnit 
        aCompartmentDic[ 'Outside' ]          =  anOutside 
        aCompartmentDic[ 'Constant' ]         =  aConstant 

        LIST.append( aCompartmentDic )

    return LIST


def get_Event_list( aSBMLmodel, timeSymbol ):
    " [[ Id , Name , StringTrigger , StringDelay , TimeUnit , [[ VariableAssignment , StringAssignment ]] ]] "
    LIST = []
    if aSBMLmodel.getEvent(0):
        NumEvent = aSBMLmodel.getNumEvents()
        for Num_Ev in range( NumEvent ):
            aEventDic = {}
            
            anEvent = aSBMLmodel.getEvent( Num_Ev )
            
            aEventDic[ 'Id' ]   = anEvent.getId()
            aEventDic[ 'Name' ] = anEvent.getName()

            aNode_Ev_Tr = anEvent.getTrigger()
            if aNode_Ev_Tr is not None and aNode_Ev_Tr.isSetMath():
                aEventDic[ 'Trigger' ] = aNode_Ev_Tr.getMath()
            else:
                aEventDic[ 'Trigger' ] = None

            aNode_Ev_De = anEvent.getDelay()
            if aNode_Ev_De is not None and aNode_Ev_De.isSetMath():
                aEventDic[ 'Delay' ] = aNode_Ev_De.getMath()
            else:
                aEventDic[ 'Delay' ] = libsbml.parseFormula( '0.0' )

            aEventDic[ 'Unit' ] = anEvent.getTimeUnits()
            
            ListOfEventAssignments = []
            if anEvent.getEventAssignment(0):
                NumEventAssignment = anEvent.getNumEventAssignments()
                for Num_Ev_As in range( NumEventAssignment ):
                    anEventAssignmentDic = {}
                    
                    anEventAssignment = anEvent.getEventAssignment( Num_Ev_As )
                    
                    anEventAssignmentDic[ 'Variable' ] = anEventAssignment.getVariable()

                    if anEventAssignment.isSetMath():
                        anEventAssignmentDic[ 'Math' ]     = anEventAssignment.getMath()

                    ListOfEventAssignments.append( anEventAssignmentDic )

            aEventDic[ 'EventAssignments' ] =  ListOfEventAssignments 
            
            LIST.append( aEventDic )

    return LIST


def get_FunctionDefinition_list( aSBMLmodel, timeSymbol ):
    " [[ Id , Name , Math, Formula ]] "
    LIST = []
    if aSBMLmodel.getFunctionDefinition(0):
        NumFunctionDefinition = aSBMLmodel.getNumFunctionDefinitions()
        for Num_FD in range( NumFunctionDefinition ):
            aFunctionDefinitionDic = {}

            aFunctionDefinition = aSBMLmodel.getFunctionDefinition( Num_FD )

            aFunctionDefinitionDic[ 'Id' ]   = aFunctionDefinition.getId()
            aFunctionDefinitionDic[ 'Name' ] = aFunctionDefinition.getName()

            if aFunctionDefinition.isSetMath():
                aFunctionDefinitionDic[ 'Math' ]    = aFunctionDefinition.getMath()
                aFunctionDefinitionDic[ 'Formula' ] = math_to_string( aFunctionDefinitionDic[ 'Math' ], timeSymbol )
            else:
                aFunctionDefinitionDic[ 'Math' ]    = None
                aFunctionDefinitionDic[ 'Formula' ] = None

            LIST.append( aFunctionDefinitionDic )

    return LIST


def get_Parameter_list( aSBMLmodel, DerivedValueDic ):
    " [[ Id , Name , Value , Unit , Constant ]] "
    LIST = []
    if aSBMLmodel.getParameter(0):
        NumParameter = aSBMLmodel.getNumParameters()
        for Num_Pa in range( NumParameter ):
            aParameterDic = {}

            aParameter = aSBMLmodel.getParameter( Num_Pa )

            anId_Pa = aParameter.getId()
            aName_Pa = aParameter.getName()
            
            """ ## -------------------------------------------------------- 
            SBML file sometime allows inappropriate initial value(s).
            Thus it's more safety to assignment the initial value using 
            AssignmentRule when it is available than to use the 
            described initial value as is.
            """ ## -------------------------------------------------------- 
            
            if aSBMLmodel.getAssignmentRule( anId_Pa ) == None:
                if aParameter.isSetValue():
                    aValue_Pa = aParameter.getValue()
                else:
                    raise TypeError, 'Initial value of %s can not been determined.' % anId_Pa
            else:
                if get_initial_value_from_AssignmentRule( aSBMLmodel, anId_Pa, DerivedValueDic ):
                    aValue_Pa = DerivedValueDic[ anId_Pa ]
                else:
                    raise TypeError, 'Initial value of %s can not been determined.' % anId_Pa
                
            anUnit_Pa = aParameter.getUnits()
            aConstant_Pa = aParameter.getConstant()

            aParameterDic[ 'Id' ]       =  anId_Pa 
            aParameterDic[ 'Name' ]     =  aName_Pa 
            aParameterDic[ 'Value' ]    =  aValue_Pa 
            aParameterDic[ 'Unit' ]     =  anUnit_Pa 
            aParameterDic[ 'Constant' ] =  aConstant_Pa 

            LIST.append( aParameterDic )

    return LIST


def get_Reaction_list( aSBMLmodel, aSBMLDocument, timeSymbol ):
    " [[ Id , Name , [ Formula , String , TimeUnit , SubstanceUnit , [[ ParameterId , ParameterName , ParameterValue , ParameterUnit , ParameterConstant ]] ] , Reversible , Fast , [[ ReactantSpecies , ( ReactantStoichiometry , ReactantStoichiometryMath ) , ReactantDenominator  ]] , [[  ProductSpecies , ( ProductStoichiometry , ProductStoichiometryMath ) , ProductDenominator ]] , [[ ModifierSpecies ]] ]] "
    LIST = []
    if aSBMLmodel.getReaction(0):
        NumReaction = aSBMLmodel.getNumReactions()
        for Num in range( NumReaction ):
            aReactionDic = {}
            aReaction = aSBMLmodel.getReaction( Num )

            anId = aReaction.getId()
            aName =aReaction.getName()

#----------KineticLaw----------------------------------
            aKineticLawDic = {}
            if aReaction.isSetKineticLaw():

                aKineticLaw = aReaction.getKineticLaw()
                #            anASTNode = libsbml.ASTNode()
                if aKineticLaw != []:

                    ## aRuleDic[ 'Formula' ], aRuleDic[ 'Math' ], aRuleDic[ 'isDiscontinuous' ]

                    if aKineticLaw.isSetFormula():
                        aKineticLawDic[ 'Math' ]            = aKineticLaw.getMath()
                        aKineticLawDic[ 'Formula' ]         = math_to_string( aKineticLawDic[ 'Math' ], timeSymbol )
                        aKineticLawDic[ 'isDiscontinuous' ] = is_discontinuous( aKineticLawDic[ 'Math' ] )
                    else:
                        aKineticLawDic[ 'Math' ]            = None
                        aKineticLawDic[ 'Formula' ]         = None
                        aKineticLawDic[ 'isDiscontinuous' ] = False

                    aKineticLawDic[ 'TimeUnit' ]      = aKineticLaw.getTimeUnits()
                    aKineticLawDic[ 'SubstanceUnit' ] = aKineticLaw.getSubstanceUnits()
            
                    if aKineticLaw.getParameter(0):
                        ListOfParameters = []
                        NumParameter_KL = aKineticLaw.getNumParameters()
                        for NumPara in range( NumParameter_KL ):
                            aParameterDic = {}
                            aParameter = aKineticLaw.getParameter( NumPara )

                            anId_KL_P      = aParameter.getId()
                            aName_KL_P     = aParameter.getName()
                            aValue_KL_P    = str( aParameter.getValue() )
                            aUnit_KL_P     = aParameter.getUnits()
                            aConstant_KL_P = aParameter.getConstant()

                            aParameterDic[ 'Id' ]       =  anId_KL_P 
                            aParameterDic[ 'Name' ]     =  aName_KL_P 
                            aParameterDic[ 'Value' ]    =  aValue_KL_P 
                            aParameterDic[ 'Unit' ]     =  aUnit_KL_P 
                            aParameterDic[ 'Constant' ] =  aConstant_KL_P 

##                            print 'Type( Math )    = %s' % type( aKineticLawDic[ 'Math' ] )
##                            print 'Type( Formula ) = %s' % type( aKineticLawDic[ 'Formula' ] )

                            ListOfParameters.append( aParameterDic )
                    else:
                        ListOfParameters = []

                    anExpressionAnnotation = aKineticLaw.getAnnotation()

                    aKineticLawDic[ 'Parameters' ]           = ListOfParameters
                    aKineticLawDic[ 'ExpressionAnnotation' ] = anExpressionAnnotation

#---------------------------------------------------------


            aReversible = aReaction.getReversible()
            aFast = aReaction.getFast()

            if( aSBMLDocument.getLevel() == 1 ):
                anIDString = 'Name'
            else:
                anIDString = 'Id'

            ListOfReactants = []
            if aReaction.getReactant(0):
                NumReactant = aReaction.getNumReactants()
                for NumR in range( NumReactant ):
                    aReactantDic = {}

                    aSpeciesReference= aReaction.getReactant( NumR )

                    aSpecies_R = aSpeciesReference.getSpecies()
                    aStoichiometry_R = aSpeciesReference.getStoichiometry()

                    aStoichiometryMath_R = None
                    if aSpeciesReference.isSetStoichiometryMath():
                        aNode_R = aSpeciesReference.getStoichiometryMath()
                        if aNode_R.isSetMath():
##                            aStoichiometryMath_R = postprocess_math_string( libsbml.formulaToString( preprocess_math_tree( aNode_R.getMath(), timeSymbol ) ), timeSymbol )
                            aStoichiometryMath_R = aNode_R.getMath()

                    aDenominator_R = aSpeciesReference.getDenominator()

                    aReactantDic[ anIDString ]          = aSpecies_R
                    aReactantDic[ 'Stoichiometry' ]     = aStoichiometry_R
                    aReactantDic[ 'StoichiometryMath' ] = aStoichiometryMath_R
                    aReactantDic[ 'Denominator' ]       = aDenominator_R

                    ListOfReactants.append( aReactantDic )


            ListOfProducts = []
            if aReaction.getProduct(0):
                NumProduct = aReaction.getNumProducts()
                for NumP in range( NumProduct ):
                    aProductDic = {}

                    aSpeciesReference = aReaction.getProduct( NumP )

                    aSpecies_P = aSpeciesReference.getSpecies()
                    aStoichiometry_P = aSpeciesReference.getStoichiometry()

                    aStoichiometryMath_P = None
                    if aSpeciesReference.isSetStoichiometryMath():
                        aNode_P = aSpeciesReference.getStoichiometryMath()
                        if aNode_P.isSetMath():
##                            aStoichiometryMath_P = postprocess_math_string( libsbml.formulaToString( preprocess_math_tree( aNode_P.getMath(), timeSymbol ) ), timeSymbol )
                            aStoichiometryMath_P = aNode_P.getMath()

                    aDenominator_P = aSpeciesReference.getDenominator()

                    aProductDic[ anIDString ]          = aSpecies_P
                    aProductDic[ 'Stoichiometry' ]     = aStoichiometry_P
                    aProductDic[ 'StoichiometryMath' ] = aStoichiometryMath_P
                    aProductDic[ 'Denominator' ]       = aDenominator_P

                    ListOfProducts.append( aProductDic )


            ListOfModifiers = []
            if aReaction.getModifier(0):
                NumModifier = aReaction.getNumModifiers()
                for NumM in range( NumModifier ):
                    aSpeciesReference = aReaction.getModifier( NumM )

                    aSpecies_M = aSpeciesReference.getSpecies()
                    ListOfModifiers.append( aSpecies_M )

            aReactionDic[ 'Id' ]                 =  anId 
            aReactionDic[ 'Name' ]               =  aName 
            aReactionDic[ 'KineticLaw' ]         =  aKineticLawDic 
            aReactionDic[ 'Reversible' ]         =  aReversible 
            aReactionDic[ 'Fast' ]               =  aFast 
            aReactionDic[ 'Reactants' ]          =  ListOfReactants 
            aReactionDic[ 'Products' ]           =  ListOfProducts 
            aReactionDic[ 'Modifiers' ]          =  ListOfModifiers 
            
            aReactionDic[ 'CommonDemoninator' ]  =  get_common_denominator( aReactionDic )
            
            LIST.append( aReactionDic )

    return LIST


def get_Rule_list( aSBMLmodel, timeSymbol ):
    " [[ RuleType, Formula, Variable ]] "
    LIST = []
    if aSBMLmodel.getRule(0):
        NumRule = aSBMLmodel.getNumRules()
        for Num in range( NumRule ):
            aRuleDic = {}
            aRule = aSBMLmodel.getRule( Num )
            
            ## aRuleDic[ 'Type' ]
            
            aRuleDic[ 'Type' ] = aRule.getTypeCode()
            
            ## aRuleDic[ 'Formula' ], aRuleDic[ 'Math' ], aRuleDic[ 'isDiscontinuous' ]
            
            if aRule.isSetFormula():
                aRuleDic[ 'Math' ] = aRule.getMath()
                aRuleDic[ 'Formula' ]         = math_to_string( aRuleDic[ 'Math' ], timeSymbol )
                aRuleDic[ 'isDiscontinuous' ] = is_discontinuous( aRuleDic[ 'Math' ] )

            else:
                aKineticLawDic[ 'Math' ] = None
                aRuleDic[ 'Formula' ] = None
                aRuleDic[ 'isDiscontinuous' ] = False
            
            ## aRuleDic[ 'Variable' ]
            
            if ( aRuleDic[ 'Type' ] == libsbml.SBML_ALGEBRAIC_RULE ):
                aRuleDic[ 'Variable' ] = ''

            elif ( aRuleDic[ 'Type' ] in ( libsbml.SBML_ASSIGNMENT_RULE, libsbml.SBML_RATE_RULE )):
                aRuleDic[ 'Variable' ] = aRule.getVariable()

            elif ( aRuleDic[ 'Type' ] == libsbml.SBML_SPECIES_CONCENTRATION_RULE ):
                aRuleDic[ 'Variable' ] = aRule.getSpecies()

            elif( aRuleDic[ 'Type' ] == libsbml.SBML_COMPARTMENT_VOLUME_RULE ):
                aRuleDic[ 'Variable' ] = aRule.getCompartment()

            elif( aRuleDic[ 'Type' ] == libsbml.SBML_PARAMETER_RULE ):
                aRuleDic[ 'Variable' ] = aRule.getName()
                
            else:
                raise TypeError, " The type of Rule must be Algebraic, Assignment or Rate Rule"

##            aRuleDic[ 'hasDelay' ]

            LIST.append( aRuleDic )

    return LIST


def get_Species_list( aSBMLmodel, DerivedValueDic ):
    " [[ Id , Name , Compartment , InitialAmount , InitialConcentration , SubstanceUnit , SpatialSizeUnit , Unit , HasOnlySubstanceUnit , BoundaryCondition , Charge , Constant ]] "
    LIST = []
    if aSBMLmodel.getSpecies(0):
        NumSpecies = aSBMLmodel.getNumSpecies()
        for Num in range( NumSpecies ):
            aSpeciesDic = {}
            aSpecies = aSBMLmodel.getSpecies( Num )

            anId_Sp = aSpecies.getId()
            aName_Sp = aSpecies.getName()
            aCompartment_Sp = aSpecies.getCompartment()

            if aSpecies.isSetInitialAmount():
                anInitialAmount_Sp = aSpecies.getInitialAmount()
            else:
                anInitialAmount_Sp = "Unknown"

            if aSpecies.isSetInitialConcentration():
                anInitialConcentration_Sp = aSpecies.getInitialConcentration()
            else:
                anInitialConcentration_Sp = "Unknown"
                

            if (( anInitialAmount_Sp == "Unknown" ) and ( anInitialConcentration_Sp == "Unknown" )):
                if get_initial_value_from_AssignmentRule( aSBMLmodel, anId_Sp, DerivedValueDic ):
                    anInitialAmount_Sp = DerivedValueDic[ anId_Sp ]

            aSubstanceUnit_Sp = aSpecies.getSubstanceUnits()
            aSpatialSizeUnit_Sp = aSpecies.getSpatialSizeUnits()
            anUnit_Sp = aSpecies.getUnits()
            aHasOnlySubstanceUnit_Sp = aSpecies.getHasOnlySubstanceUnits()
            aBoundaryCondition_Sp = aSpecies.getBoundaryCondition()
            aCharge_Sp = aSpecies.getCharge()
            aConstant_Sp = aSpecies.getConstant()


            aSpeciesDic[ 'Id' ]                   =  anId_Sp 
            aSpeciesDic[ 'Name' ]                 =  aName_Sp 
            aSpeciesDic[ 'Compartment' ]          =  aCompartment_Sp 
            aSpeciesDic[ 'InitialAmount' ]        =  anInitialAmount_Sp 
            aSpeciesDic[ 'InitialConcentration' ] =  anInitialConcentration_Sp 
            aSpeciesDic[ 'SubstanceUnit' ]        =  aSubstanceUnit_Sp 
            aSpeciesDic[ 'SpatialSizeUnit' ]      =  aSpatialSizeUnit_Sp 
            aSpeciesDic[ 'Unit' ]                 =  anUnit_Sp 
            aSpeciesDic[ 'HasOnlySubstanceUnit' ] =  aHasOnlySubstanceUnit_Sp 
            aSpeciesDic[ 'BoundaryCondition' ]    =  aBoundaryCondition_Sp 
            aSpeciesDic[ 'Charge' ]               =  aCharge_Sp 
            aSpeciesDic[ 'Constant' ]             =  aConstant_Sp 
       
            LIST.append( aSpeciesDic )

    return LIST


def get_UnitDefinition_list( aSBMLmodel ):
    " [[ Id , Name , [[ Kind , Exponent , Scale , Multiplier , Offset ]] ]] "
    LIST = []
    if aSBMLmodel.getUnitDefinition(0):
        NumUnitDefinition = aSBMLmodel.getNumUnitDefinitions()
        for Num1 in range( NumUnitDefinition ):
            aUnitDefinitionDic = {}

            anUnitDefinition = aSBMLmodel.getUnitDefinition( Num1 )

            anId = anUnitDefinition.getId()
            aName = anUnitDefinition.getName()

            ListOfUnits = []
            if anUnitDefinition.getUnit(0):
                NumUnit = anUnitDefinition.getNumUnits()
                for Num2 in range( NumUnit ):
                    anUnitDic = {}
                    anUnit = anUnitDefinition.getUnit( Num2 )

                    anUnitKind = anUnit.getKind()

                    aKind = libsbml.UnitKind_toString( anUnitKind )
                    anExponent = anUnit.getExponent()
                    aScale = anUnit.getScale()
                    aMultiplier = anUnit.getMultiplier()
                    anOffset = anUnit.getOffset()

                    anUnitDic[ 'Kind' ]       = aKind
                    anUnitDic[ 'Exponent' ]   = anExponent
                    anUnitDic[ 'Scale' ]      = aScale
                    anUnitDic[ 'Multiplier' ] = aMultiplier
                    anUnitDic[ 'Offset' ]     = anOffset

                    ListOfUnits.append( anUnitDic )

            aUnitDefinitionDic[ 'Id' ]         =  anId 
            aUnitDefinitionDic[ 'Name' ]       =  aName 
            aUnitDefinitionDic[ 'Definition' ] =  ListOfUnits 

                
            LIST.append( aUnitDefinitionDic )

    return LIST


# --------------------------------------------------
#  Get a common denominator of a reaction
# --------------------------------------------------

def get_common_denominator( aReaction ):

    coefficientList = []
    
    for aReactant in aReaction[ 'Reactants' ]:
        if aReactant[ 'Stoichiometry' ] != 0:
            coefficientList.append( fractions.Fraction( decimal.Decimal.from_float( aReactant[ 'Stoichiometry' ] )))
    
    for aProduct in aReaction[ 'Products' ]:
        if aProduct[ 'Stoichiometry' ] != 0:
            coefficientList.append( fractions.Fraction( decimal.Decimal.from_float( aProduct[ 'Stoichiometry' ] )))

    denominatorList = []
    
    for aCoefficient in coefficientList:
        denominatorList.append( aCoefficient.denominator )

    aGCD = decimal.Decimal( 1 )
    aLCM = decimal.Decimal( 1 )

    for aCoefficient in coefficientList:
        aGCD = fractions.gcd( aGCD, aCoefficient.denominator )
        aLCM = aLCM * aCoefficient.denominator / aGCD

#    print 'Coefficient: %s' % coefficientList
#    print 'LCM: %s' % aLCM

    return float( aLCM )


# --------------------------------------------------
#  Check the Continuity of a Formula
# --------------------------------------------------

def is_discontinuous( aMathNode ):

    discontinuousTypes = (
        libsbml.AST_FUNCTION_DELAY,
        libsbml.AST_FUNCTION_PIECEWISE,
        libsbml.AST_LOGICAL_AND,
        libsbml.AST_LOGICAL_NOT,
        libsbml.AST_LOGICAL_OR,
        libsbml.AST_LOGICAL_XOR,
        libsbml.AST_NAME_TIME,
        libsbml.AST_RELATIONAL_EQ,
        libsbml.AST_RELATIONAL_GEQ,
        libsbml.AST_RELATIONAL_GT,
        libsbml.AST_RELATIONAL_LEQ,
        libsbml.AST_RELATIONAL_LT,
        libsbml.AST_RELATIONAL_NEQ )

    if aMathNode.getType() in discontinuousTypes:
        return True
    
    numChildren = aMathNode.getNumChildren()
    if numChildren != 0:
        for i in range( numChildren ):
            if is_discontinuous( aMathNode.getChild( i ) ):
                return True
    
    return False


# --------------------------------------------------
#  Pre/Post-process of Formula
# --------------------------------------------------

def get_TimeSymbol( aSBMLmodel ):

    timeSymbol = "time"

    while ( aSBMLmodel.getElementBySId( timeSymbol ) ):
        timeSymbol = "_%s" % timeSymbol

##    print "Time Symbol =\"%s\"" % timeSymbol
    return timeSymbol


def preprocess_math_tree( aNode, timeSymbol ):

    # Set AST_NAME_TIME node's name (SBML does not guarantee its uniqueness)
    if ( aNode.getType() == libsbml.AST_NAME_TIME ):
        aNode.setName( timeSymbol )

    for i in range( aNode.getNumChildren() ):
        preprocess_math_tree( aNode.getChild( i ), timeSymbol )

    return aNode


def postprocess_math_string( aFormula, timeSymbol ):

    ## It's OK if timeSymbol = "time"
    aFormula = re.sub( r"(\W)%s(\W)" % timeSymbol, r'\1<t>\2', aFormula )
    aFormula = re.sub( r"^%s(\W)" % timeSymbol, r'<t>\1', aFormula )
    aFormula = re.sub( r"(\W)%s$" % timeSymbol, r'\1<t>', aFormula )
    aFormula = re.sub( r"^%s$" % timeSymbol, r'<t>', aFormula )

##    print 'Formula: %s' % aFormula

    return aFormula


def math_to_string( AST_node, time_symbol ):
    return postprocess_math_string( libsbml.formulaToString( preprocess_math_tree( AST_node, time_symbol ) ), time_symbol )

# --------------------------------------------------
#  Computation of Initial Value
# --------------------------------------------------

def get_initial_value_from_AssignmentRule( aSBMLmodel, aVariableID, DerivedValueDic ):
    '''
    return False when the initial value can not calculate.
    '''

    if ( aVariableID in DerivedValueDic ):
        return DerivedValueDic[ aVariableID ]

    anAssignmentRule = aSBMLmodel.getAssignmentRule( aVariableID )
    if ( anAssignmentRule == None ):
        return False                                                   ### FIXME

    aFormulaTree = anAssignmentRule.getMath().deepCopy()

##    print "\nAssignmentRule for %s" % aVariableID
##    print "  %s\n" % postprocess_math_string( libsbml.formulaToString( preprocess_math_tree( anAssignmentRule.getMath(), get_TimeSymbol( aSBMLmodel ) ) ), get_TimeSymbol( aSBMLmodel ) )
##    print "Initial Construction:\n"
##    dump_tree_construction_of_AST_node( aFormulaTree )
##    print '\n'

    aCounter = 1
    while ( aCounter > 0 ):
        aCounter = 0
        _convert_AST_NAME_to_value( aSBMLmodel, aFormulaTree, aCounter, DerivedValueDic )
##        if aCounter == 0:
##            print 'Initial Value: %s = %f' % ( aVariableID, aSBMLmodel.getElementBySId( aVariableID ).getValue() )

##    print "Name replaced with value:\n"
##    dump_tree_construction_of_AST_node( aFormulaTree )
##    print '\n'

##    dump_tree_construction_of_AST_node( aFormulaTree )

    while ( aFormulaTree.getNumChildren() > 0 ):
        _calc_AST_node_value( aFormulaTree )

##    print "Initial Value of %s: %s" % ( aVariableID, _get_AST_node_value( aFormulaTree ))

    DerivedValueDic[ aVariableID ] = _get_AST_node_value( aFormulaTree )
    return True


def _convert_AST_NAME_to_value( aSBMLmodel, aNode, aCounter, DerivedValueDic ):

    if ( aNode.getType() == libsbml.AST_NAME ):
        aVariableID = aNode.getName()
        anElement = aSBMLmodel.getElementBySId( aVariableID )
        anElementType = anElement.getTypeCode()

        ## Constants
        
        if ( anElementType == libsbml.AST_CONSTANT_E ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.e )
        
        elif ( anElementType == libsbml.AST_CONSTANT_PI ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.pi )
        
        elif ( anElementType == libsbml.AST_CONSTANT_FALSE ):
            aNode.setType( libsbml.AST_INTEGER )
            aNode.setValue( 0 )
        
        elif ( anElementType == libsbml.AST_CONSTANT_TRUE ):
            aNode.setType( libsbml.AST_INTEGER )
            aNode.setValue( 1 )
        
        elif ( anElementType == libsbml.AST_NAME_AVOGADRO ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( 6.0221367e+23 )
        
        ## Variables
        
        elif ( aVariableID in DerivedValueDic ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( DerivedValueDic[ aVariableID ] )

        elif ( anElementType == libsbml.SBML_SPECIES ):
            if anElement.isSetInitialAmount():
                aNode.setType( libsbml.AST_REAL )
                aNode.setValue( anElement.getInitialAmount() )
            elif anElement.isSetInitialConcentration():
                aNode.setType( libsbml.AST_REAL )
                aNode.setValue( anElement.getInitialConcentration() * aSBMLmodel.getCompartment( anElement.getCompartment() ).getSize() )
            else:
                if get_initial_value_from_AssignmentRule( aSBMLmodel, aVariableID, DerivedValueDic ):
                    aNode.setType( libsbml.AST_REAL )
                    aNode.setValue( DerivedValueDic[ aVariableID ] )
                else:
                    aCounter += 1
        
        elif ( anElementType == libsbml.SBML_PARAMETER ):
            if anElement.isSetValue() == True:
                aNode.setType( libsbml.AST_REAL )
                aNode.setValue( anElement.getValue() )
            else:
                if get_initial_value_from_AssignmentRule( aSBMLmodel, aVariableID, DerivedValueDic ):
                    aNode.setType( libsbml.AST_REAL )
                    aNode.setValue( DerivedValueDic[ aVariableID ] )
                else:
                    aCounter += 1
        
        elif ( anElementType == libsbml.SBML_COMPARTMENT ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( anElement.getSize() )
        
        else:
            raise TypeError,\
            "_convert_AST_NAME_to_value: Element type must be Species, Parameter, or Compartment"

    ##     Time ( initial value = 0.0 )

    elif ( aNode.getType() == libsbml.AST_NAME_TIME ):
        aNode.setType( libsbml.AST_REAL )
        aNode.setValue( 0.0 )

    for i in range( aNode.getNumChildren() ):
        _convert_AST_NAME_to_value( aSBMLmodel, aNode.getChild( i ), aCounter, DerivedValueDic )


def _calc_AST_node_value( aNode ):
    if ( sys.version_info[0] >= 3 or
         ( sys.version_info[0] == 2 and sys.version_info[1] >= 6 )):
        geqPython26 = True
    else:
        geqPython26 = False

    isReady = True    ## True:  All children nodes are Number

    for i in range( aNode.getNumChildren() ):
        aChild = aNode.getChild( i )
        if ( aChild.isNumber() == False ):
            isReady = False
            break

    if ( isReady == True ):
        aNodeType = aNode.getType()
        
        if   ( aNodeType == libsbml.AST_RATIONAL ):
            aRealValue = aNode.getReal()
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( aRealValue )
        elif ( aNodeType == libsbml.AST_DIVIDE ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( float( _get_AST_node_value( aNode.getLeftChild())) / float( _get_AST_node_value( aNode.getRightChild())) )

        elif( aNodeType == libsbml.AST_MINUS ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( float( _get_AST_node_value( aNode.getLeftChild())) - float( _get_AST_node_value( aNode.getRightChild())) )

        elif( aNodeType == libsbml.AST_PLUS ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( float( _get_AST_node_value( aNode.getLeftChild())) + float( _get_AST_node_value( aNode.getRightChild())) )

        elif ( aNodeType == libsbml.AST_TIMES ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( float( _get_AST_node_value( aNode.getLeftChild())) * float( _get_AST_node_value( aNode.getRightChild())) )

        elif ( aNodeType == libsbml.AST_POWER ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( pow( float( _get_AST_node_value( aNode.getLeftChild())), float( _get_AST_node_value( aNode.getRightChild()))) )

        ## Logical operators

        elif ( aNodeType == libsbml.AST_LOGICAL_AND ):
            aNode.setType( libsbml.AST_INTEGER )
            aNode.setValue( _get_AST_node_value( aNode.getLeftChild() ) and _get_AST_node_value( aNode.getRightChild() ) )

        elif ( aNodeType == libsbml.AST_LOGICAL_NOT ):
            aNode.setType( libsbml.AST_INTEGER )
            aNode.setValue( not _get_AST_node_value( aNode.getChild( 0 )))

        elif ( aNodeType == libsbml.AST_LOGICAL_OR ):
            aNode.setType( libsbml.AST_INTEGER )
            aNode.setValue( _get_AST_node_value( aNode.getLeftChild() ) or _get_AST_node_value( aNode.getRightChild() ) )

        elif ( aNodeType == libsbml.AST_LOGICAL_XOR ):
            aNode.setType( libsbml.AST_INTEGER )
            aNode.setValue( ( not( _get_AST_node_value( aNode.getLeftChild() ))) and ( not( _get_AST_node_value( aNode.getRightChild() ))))

        ## Relational operators

        elif ( aNodeType == libsbml.AST_RELATIONAL_EQ ):
            aNode.setType( libsbml.AST_INTEGER )
            if ( float( _get_AST_node_value( aNode.getLeftChild() )) == float( _get_AST_node_value( aNode.getRightChild() )) ):
                aNode.setValue( 1 )
            else:
                aNode.setValue( 0 )

        elif ( aNodeType == libsbml.AST_RELATIONAL_GEQ ):
            aNode.setType( libsbml.AST_INTEGER )
            if ( float( _get_AST_node_value( aNode.getLeftChild() )) >= float( _get_AST_node_value( aNode.getRightChild() )) ):
                aNode.setValue( 1 )
            else:
                aNode.setValue( 0 )

        elif ( aNodeType == libsbml.AST_RELATIONAL_GT ):
            aNode.setType( libsbml.AST_INTEGER )
            if ( float( _get_AST_node_value( aNode.getLeftChild() )) > float( _get_AST_node_value( aNode.getRightChild() )) ):
                aNode.setValue( 1 )
            else:
                aNode.setValue( 0 )

        elif ( aNodeType == libsbml.AST_RELATIONAL_LEQ ):
            aNode.setType( libsbml.AST_INTEGER )
            if ( float( _get_AST_node_value( aNode.getLeftChild() )) <= float( _get_AST_node_value( aNode.getRightChild() )) ):
                aNode.setValue( 1 )
            else:
                aNode.setValue( 0 )

        elif ( aNodeType == libsbml.AST_RELATIONAL_LT ):
            aNode.setType( libsbml.AST_INTEGER )
            if ( float( _get_AST_node_value( aNode.getLeftChild() )) < float( _get_AST_node_value( aNode.getRightChild() )) ):
                aNode.setValue( 1 )
            else:
                aNode.setValue( 0 )

        elif ( aNodeType == libsbml.AST_RELATIONAL_NEQ ):
            aNode.setType( libsbml.AST_INTEGER )
            if ( float( _get_AST_node_value( aNode.getLeftChild() )) != float( _get_AST_node_value( aNode.getRightChild() )) ):
                aNode.setValue( 1 )
            else:
                aNode.setValue( 0 )

        ## Functions

        ##     Power and logarithmic functions

        elif ( aNodeType == libsbml.AST_FUNCTION_EXP ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.exp( float( _get_AST_node_value( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_LN ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.log( float( _get_AST_node_value( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_LOG ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.log( float( _get_AST_node_value( aNode.getLeftChild())), float( _get_AST_node_value( aNode.getRightChild() ))))

        elif ( aNodeType == libsbml.AST_FUNCTION_POWER ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( pow( float( _get_AST_node_value( aNode.getLeftChild())), float( _get_AST_node_value( aNode.getRightChild() ))))

        elif ( aNodeType == libsbml.AST_FUNCTION_ROOT ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( pow( float( _get_AST_node_value( aNode.getLeftChild())), 1.0 / float( _get_AST_node_value( aNode.getRightChild() ))))

        ##     Number-theoretic and representation functions

        elif ( aNodeType == libsbml.AST_FUNCTION_ABS ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( abs( float( _get_AST_node_value( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_CEILING ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.ceil( float( _get_AST_node_value( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_FACTORIAL ):
            aNode.setType( libsbml.AST_INTEGER )
            aNode.setValue( math.factorial( _get_AST_node_value( aNode.getChild( 0 ))))

        elif ( aNodeType == libsbml.AST_FUNCTION_FLOOR ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.floor( float( _get_AST_node_value( aNode.getChild( 0 )))))

        ##     Trigonometric functions

        elif ( aNodeType == libsbml.AST_FUNCTION_SIN ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.sin( float( _get_AST_node_value( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_COS ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.cos( float( _get_AST_node_value( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_TAN ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.tan( float( _get_AST_node_value( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_CSC ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( 1.0 / math.sin( float( _get_AST_node_value( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_SEC ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( 1.0 / math.cos( float( _get_AST_node_value( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_COT ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( 1.0 / math.tan( float( _get_AST_node_value( aNode.getChild( 0 )))))

        ##     Inverse trigonometric functions

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCSIN ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.asin( float( _get_AST_node_value( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCCOS ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.acos( float( _get_AST_node_value( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCTAN ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.atan( float( _get_AST_node_value( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCCSC ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( 1.0 / math.asin( float( _get_AST_node_value( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCSEC ):  ## arc-secant
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.acos( 1.0 / float( _get_AST_node_value( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCCOT ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.atan( 1.0 / float( _get_AST_node_value( aNode.getChild( 0 )))))

        ##     Hyperbolic functions

        elif ( aNodeType == libsbml.AST_FUNCTION_SINH ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.sinh( float( _get_AST_node_value( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_COSH ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.cosh( float( _get_AST_node_value( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_TANH ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.tanh( float( _get_AST_node_value( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_CSCH ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( 1.0 / math.sinh( float( _get_AST_node_value( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_SECH ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( 1.0 / math.cosh( float( _get_AST_node_value( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_COTH ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( 1.0 / math.tanh( float( _get_AST_node_value( aNode.getChild( 0 )))))

        ##     Inverse hyperbolic functions

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCSINH ):
            aNode.setType( libsbml.AST_REAL )
            if geqPython26:
                aNode.setValue( math.asinh( float( _get_AST_node_value( aNode.getChild( 0 )))))
            else:
                x = float( _get_AST_node_value( aNode.getChild( 0 )))
                aNode.setValue( math.log( x + math.sqrt( x * x + 1.0 )))

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCCOSH ):
            aNode.setType( libsbml.AST_REAL )
            if geqPython26:
                aNode.setValue( math.acosh( float( _get_AST_node_value( aNode.getChild( 0 )))))
            else:
                x = float( _get_AST_node_value( aNode.getChild( 0 )))
                aNode.setValue( math.log( x + math.sqrt( x + 1.0 ) * math.sqrt( x - 1.0 )))

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCTANH ):
            aNode.setType( libsbml.AST_REAL )
            if geqPython26:
                aNode.setValue( math.atanh( float( _get_AST_node_value( aNode.getChild( 0 )))))
            else:
                x = float( _get_AST_node_value( aNode.getChild( 0 )))
                aNode.setValue( 0.5 * math.log( ( 1.0 + x ) / ( 1.0 - x )))

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCCSCH ):
            aNode.setType( libsbml.AST_REAL )
            if geqPython26:
                aNode.setValue( math.asinh( 1.0 / float( _get_AST_node_value( aNode.getChild( 0 )))))
            else:
                x = float( _get_AST_node_value( aNode.getChild( 0 )))
                aNode.setValue( math.log( 1.0 / x + math.sqrt( 1.0 / x / x + 1.0 )))

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCSECH ):
            aNode.setType( libsbml.AST_REAL )
            if geqPython26:
                aNode.setValue( math.acosh( 1.0 / float( _get_AST_node_value( aNode.getChild( 0 )))))
            else:
                x = float( _get_AST_node_value( aNode.getChild( 0 )))
                aNode.setValue( math.log( 1.0 / x + math.sqrt( 1.0 / x + 1.0 ) * math.sqrt( 1.0 / x - 1.0 )))

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCCOTH ):
            aNode.setType( libsbml.AST_REAL )
            if geqPython26:
                aNode.setValue( math.atanh( 1.0 / float( _get_AST_node_value( aNode.getChild( 0 )))))
            else:
                x = float( _get_AST_node_value( aNode.getChild( 0 )))
                aNode.setValue( 0.5 * math.log( ( x + 1.0 ) / ( x - 1.0 )))

        ##     Other functions

        elif ( aNodeType == libsbml.AST_FUNCTION_DELAY ):
            ### At t = 0, delayed value is not available, thus value at t = 0 is used.
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( _get_AST_node_value( aNode.getChild( 0 )))

        elif ( aNodeType == libsbml.AST_FUNCTION_PIECEWISE ):
##            print "\nCulc AST_FUNCTION_PIECEWISE:"
##            dump_tree_construction_of_AST_node( aNode )
##
##            print "  #Children = %d" % aNode.getNumChildren()
            for i in range( aNode.getNumChildren() / 2 ):
                if ( _get_AST_node_value( aNode.getChild( i * 2 + 1 )) != 0 ):
##                    aNode = aNode.getChild( i * 2 ).deepCopy()
                    aNode.setType( aNode.getChild( i * 2 ).getType() )
                    aNode.setValue( _get_AST_node_value( aNode.getChild( i * 2 )))
##                    print "  piece(%d) is True! Value is %s" % ( i, _get_AST_node_value( aNode ))
                    break
            
            if ( aNode.getType() == libsbml.AST_FUNCTION_PIECEWISE ):
                if ( aNode.getNumChildren() % 2 == 1 ):
##                    aNode = aNode.getRightChild().deepCopy()
                    aNode.setType( aNode.getRightChild().getType() )
                    aNode.setValue( _get_AST_node_value( aNode.getRightChild()))
                else:
                    raise TypeError,\
                    "Can't derive an initial value from a piecewise function"

        ## Unknown

        elif ( aNodeType == libsbml.AST_UNKNOWN ):
            raise TypeError,\
            "Unknown operator is detected in Formula"

        '''
AST_FUNCTION            : Solved by converter
AST_LAMBDA              : Solved by converter
        '''

        _remove_all_Children( aNode )

    for i in range( aNode.getNumChildren() ):
        _calc_AST_node_value( aNode.getChild( i ) )


def _get_AST_node_value( aNode ):
    aNodeType = aNode.getType()
    if ( aNode.isReal() ):
        return aNode.getReal()
    elif ( aNode.isInteger() ):
        return aNode.getInteger()
    else:
        raise TypeError,\
        "aNode must be number"


def _remove_all_Children( aNode ):
    while ( aNode.getNumChildren() > 0 ):
        aNode.removeChild( 0 )
    return

# --------------------------------------------------
#  Output the Construction of Math Tree (ASTNode)
# --------------------------------------------------

def dump_tree_construction_of_AST_node( aNode ):

##    print aNode.getType()

    if ( aNode.getType() == libsbml.AST_CONSTANT_E ):
        print "  AST_CONSTANT_E"
    elif ( aNode.getType() == libsbml.AST_CONSTANT_FALSE ):
        print "  AST_CONSTANT_FALSE"
    elif ( aNode.getType() == libsbml.AST_CONSTANT_PI ):
        print "  AST_CONSTANT_PI"
    elif ( aNode.getType() == libsbml.AST_CONSTANT_TRUE ):
        print "  AST_CONSTANT_TRUE"
    elif ( aNode.getType() == libsbml.AST_DIVIDE ):
        print "  AST_DIVIDE"
    elif ( aNode.getType() == libsbml.AST_FUNCTION ):
        print "  AST_FUNCTION"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_ABS ):
        print "  AST_FUNCTION_ABS"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_ARCCOS ):
        print "  AST_FUNCTION_ARCCOS"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_ARCCOSH ):
        print "  AST_FUNCTION_ARCCOSH"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_ARCCOT ):
        print "  AST_FUNCTION_ARCCOT"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_ARCCOTH ):
        print "  AST_FUNCTION_ARCCOTH"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_ARCCSC ):
        print "  AST_FUNCTION_ARCCSC"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_ARCCSCH ):
        print "  AST_FUNCTION_ARCCSCH"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_ARCSEC ):
        print "  AST_FUNCTION_ARCSEC"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_ARCSECH ):
        print "  AST_FUNCTION_ARCSECH"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_ARCSIN ):
        print "  AST_FUNCTION_ARCSIN"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_ARCSINH ):
        print "  AST_FUNCTION_ARCSINH"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_ARCTAN ):
        print "  AST_FUNCTION_ARCTAN"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_ARCTANH ):
        print "  AST_FUNCTION_ARCTANH"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_CEILING ):
        print "  AST_FUNCTION_CEILING"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_COS ):
        print "  AST_FUNCTION_COS"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_COSH ):
        print "  AST_FUNCTION_COSH"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_COT ):
        print "  AST_FUNCTION_COT"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_COTH ):
        print "  AST_FUNCTION_COTH"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_CSC ):
        print "  AST_FUNCTION_CSC"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_CSCH ):
        print "  AST_FUNCTION_CSCH"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_DELAY ):
        print "  AST_FUNCTION_DELAY"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_EXP ):
        print "  AST_FUNCTION_EXP"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_FACTORIAL ):
        print "  AST_FUNCTION_FACTORIAL"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_FLOOR ):
        print "  AST_FUNCTION_FLOOR"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_LN ):
        print "  AST_FUNCTION_LN"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_LOG ):
        print "  AST_FUNCTION_LOG"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_PIECEWISE ):
        print "  AST_FUNCTION_PIECEWISE"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_POWER ):
        print "  AST_FUNCTION_POWER"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_ROOT ):
        print "  AST_FUNCTION_ROOT"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_SEC ):
        print "  AST_FUNCTION_SEC"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_SECH ):
        print "  AST_FUNCTION_SECH"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_SIN ):
        print "  AST_FUNCTION_SIN"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_SINH ):
        print "  AST_FUNCTION_SINH"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_TAN ):
        print "  AST_FUNCTION_TAN"
    elif ( aNode.getType() == libsbml.AST_FUNCTION_TANH ):
        print "  AST_FUNCTION_TANH"
    elif ( aNode.getType() == libsbml.AST_INTEGER ):
        print "  AST_INTEGER: %d" % aNode.getInteger()
    elif ( aNode.getType() == libsbml.AST_LAMBDA ):
        print "  AST_LAMBDA"
    elif ( aNode.getType() == libsbml.AST_LOGICAL_AND ):
        print "  AST_LOGICAL_AND"
    elif ( aNode.getType() == libsbml.AST_LOGICAL_NOT ):
        print "  AST_LOGICAL_NOT"
    elif ( aNode.getType() == libsbml.AST_LOGICAL_OR ):
        print "  AST_LOGICAL_OR"
    elif ( aNode.getType() == libsbml.AST_LOGICAL_XOR ):
        print "  AST_LOGICAL_XOR"
    elif ( aNode.getType() == libsbml.AST_MINUS ):
        print "  AST_MINUS"
    elif ( aNode.getType() == libsbml.AST_NAME ):
        print "  AST_NAME: %s" % aNode.getName()
    elif ( aNode.getType() == libsbml.AST_NAME_AVOGADRO ):
        print "  AST_NAME_AVOGADRO"
    elif ( aNode.getType() == libsbml.AST_NAME_TIME ):
        print "  AST_NAME_TIME"
    elif ( aNode.getType() == libsbml.AST_PLUS ):
        print "  AST_PLUS"
    elif ( aNode.getType() == libsbml.AST_POWER ):
        print "  AST_POWER"
    elif ( aNode.getType() == libsbml.AST_RATIONAL ):
        print "  AST_RATIONAL"
    elif ( aNode.getType() == libsbml.AST_REAL ):
        print "  AST_REAL: %f" % aNode.getReal()
    elif ( aNode.getType() == libsbml.AST_REAL_E ):
        print "  AST_REAL_E"
    elif ( aNode.getType() == libsbml.AST_RELATIONAL_EQ ):
        print "  AST_RELATIONAL_EQ"
    elif ( aNode.getType() == libsbml.AST_RELATIONAL_GEQ ):
        print "  AST_RELATIONAL_GEQ"
    elif ( aNode.getType() == libsbml.AST_RELATIONAL_GT ):
        print "  AST_RELATIONAL_GT"
    elif ( aNode.getType() == libsbml.AST_RELATIONAL_LEQ ):
        print "  AST_RELATIONAL_LEQ"
    elif ( aNode.getType() == libsbml.AST_RELATIONAL_LT ):
        print "  AST_RELATIONAL_LT"
    elif ( aNode.getType() == libsbml.AST_RELATIONAL_NEQ ):
        print "  AST_RELATIONAL_NEQ"
    elif ( aNode.getType() == libsbml.AST_TIMES ):
        print "  AST_TIMES"
    elif ( aNode.getType() == libsbml.AST_UNKNOWN ):
        print "  AST_UNKNOWN"

    for i in range( aNode.getNumChildren() ):
        dump_tree_construction_of_AST_node( aNode.getChild( i ) )


