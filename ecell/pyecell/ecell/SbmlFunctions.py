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
import math, sys

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


def getCompartment( aSBMLmodel ):
    " [[ Id , Name , SpatialDimension , Size , Volume , Unit , Ouside , Constant ]] "
    LIST = []
    theList = aSBMLmodel.getListOfCompartments()
    #if Model_getCompartment( aSBMLmodel , 0 ):
        
    NumCompartment = len( theList )
        
    for Num in range( NumCompartment ):
        ListOfCompartment = []
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

        ListOfCompartment.append( anId )
        ListOfCompartment.append( aName )
        ListOfCompartment.append( aSpatialDimension )
        ListOfCompartment.append( aSize )
        ListOfCompartment.append( aVolume )
        ListOfCompartment.append( anUnit )
        ListOfCompartment.append( anOutside )
        ListOfCompartment.append( aConstant )

        LIST.append( ListOfCompartment )

    return LIST


def getEvent( aSBMLmodel ):
    " [[ Id , Name , StringTrigger , StringDelay , TimeUnit , [[ VariableAssignment , StringAssignment ]] ]] "
    LIST = []
    if aSBMLmodel.getEvent(0):
        NumEvent = aSBMLmodel.getNumEvents()
        for Num_Ev in range( NumEvent ):
            ListOfEvent = []
            
            anEvent = aSBMLmodel.getEvent( Num_Ev )
            
            anId_Ev = anEvent.getId()
            aName_Ev = anEvent.getName()
            aString_Ev_Tr = ''
            aString_Ev_De = ''

            aNode_Ev_Tr = anEvent.getTrigger()
            if aNode_Ev_Tr is not None and aNode_Ev_Tr.isSetMath():
                aString_Ev_Tr = sub( libsbml.formulaToString,
                    aNode_Ev_Tr.getMath() )

            aNode_Ev_De = anEvent.getDelay()
            if aNode_Ev_De is not None and aNode_Ev_De.isSetMath():
                aString_Ev_De = sub( libsbml.formulaToString,
                    aNode_Ev_Tr.getMath() )

            aTimeUnit_Ev = anEvent.getTimeUnits()
            
            ListOfEventAssignments = []
            if anEvent.getEventAssignment(0):
                NumEventAssignment = anEvent.getNumEventAssignments()
                for Num_Ev_As in range( NumEventAssignment ):
                    ListOfEventAssignment = []
                    
                    anEventAssignment = anEvent.getEventAssignment( Num_Ev_As )
                    
                    aVariable_Ev_As = anEventAssignment.getVariable()
                    aString_Ev_As = ''

                    if anEventAssignment.isSetMath():
                        aString_Ev_As = sub( libsbml.formulaToString,
                                anEventAssignment.getMath() )

                    ListOfEventAssignment.append( aVariable_Ev_As )
                    ListOfEventAssignment.append( aString_Ev_As )
                    
                    ListOfEventAssignments.append( ListOfEventAssignment )

            ListOfEvent.append( anId_Ev )
            ListOfEvent.append( aName_Ev )
            ListOfEvent.append( aString_Ev_Tr )
            ListOfEvent.append( aString_Ev_De )
            ListOfEvent.append( aTimeUnit_Ev )
            ListOfEvent.append( ListOfEventAssignments )
            
            LIST.append( ListOfEvent )

    return LIST


def getFunctionDefinition( aSBMLmodel ):
    " [[ Id , Name , String ]] "
    LIST = []
    if aSBMLmodel.getFunctionDefinition(0):
        NumFunctionDefinition = aSBMLmodel.getNumFunctionDefinitions()
        for Num_FD in range( NumFunctionDefinition ):
            ListOfFunctionDefinition = []

            aFunctionDefinition = aSBMLmodel.getFunctionDefinition( Num_FD )

            anId_FD = aFunctionDefinition.getId()
            aName_FD = aFunctionDefinition.getName()
            aString_FD = ''

            if aFunctionDefinition.isSetMath():
                aString_FD = sub( libsbml.formulaToString,
                    aFunctionDefinition.getMath() )

            ListOfFunctionDefinition.append( anId_FD )
            ListOfFunctionDefinition.append( aName_FD )
            ListOfFunctionDefinition.append( aString_FD )

            LIST.append( ListOfFunctionDefinition )

    return LIST


def getParameter( aSBMLmodel, DerivedValueDic ):
    " [[ Id , Name , Value , Unit , Constant ]] "
    LIST = []
    if aSBMLmodel.getParameter(0):
        NumParameter = aSBMLmodel.getNumParameters()
        for Num_Pa in range( NumParameter ):
            ListOfParameter = []

            aParameter = aSBMLmodel.getParameter( Num_Pa )

            anId_Pa = aParameter.getId()
            aName_Pa = aParameter.getName()
            
            if aParameter.isSetValue():
                aValue_Pa = aParameter.getValue()
            else:
                if getInitialValueFromAssignmentRule( aSBMLmodel, anId_Pa, DerivedValueDic ):
                    aValue_Pa = DerivedValueDic[ anId_Pa ]
                else:
                    aValue_Pa = 'Unknown'
                
            anUnit_Pa = aParameter.getUnits()
            aConstant_Pa = aParameter.getConstant()

            ListOfParameter.append( anId_Pa )
            ListOfParameter.append( aName_Pa )
            ListOfParameter.append( aValue_Pa )
            ListOfParameter.append( anUnit_Pa )
            ListOfParameter.append( aConstant_Pa )

            LIST.append( ListOfParameter )

    return LIST


def getReaction( aSBMLmodel, aSBMLDocument ):
    " [[ Id , Name , [ Formula , String , TimeUnit , SubstanceUnit , [[ ParameterId , ParameterName , ParameterValue , ParameterUnit , ParameterConstant ]] ] , Reversible , Fast , [[ ReactantSpecies , ( ReactantStoichiometry , ReactantStoichiometryMath ) , ReactantDenominator  ]] , [[  ProductSpecies , ( ProductStoichiometry , ProductStoichiometryMath ) , ProductDenominator ]] , [[ ModifierSpecies ]] ]] "
    LIST = []
    if aSBMLmodel.getReaction(0):
        NumReaction = aSBMLmodel.getNumReactions()
        for Num in range( NumReaction ):
            ListOfReaction = []
            aReaction = aSBMLmodel.getReaction( Num )

            anId = aReaction.getId()
            aName =aReaction.getName()

#----------KineticLaw----------------------------------
            ListOfKineticLaw = []
            if aReaction.isSetKineticLaw():

                aKineticLaw = aReaction.getKineticLaw()
                #            anASTNode = libsbml.ASTNode()
                if aKineticLaw != []:

                    if aKineticLaw.isSetFormula():
                        aFormula_KL = aKineticLaw.getFormula()
                    else:
                        aFormula_KL = ''
                  
                    aMath = []
                    if( aSBMLDocument.getLevel() == 1 ):
                        aMath.append( '' )
                    else:
                        if aKineticLaw.isSetMath():
                            aMath.append(
                                libsbml.formulaToString( aKineticLaw.getMath() )
                                )
                        else:
                            aMath.append( '' )

                    aString_KL = aMath
                    
                    aTimeUnit_KL = aKineticLaw.getTimeUnits()
                    aSubstanceUnit_KL = aKineticLaw.getSubstanceUnits()
            
                    if aKineticLaw.getParameter(0):
                        ListOfParameters = []
                        NumParameter_KL = aKineticLaw.getNumParameters()
                        for NumPara in range( NumParameter_KL ):
                            ListOfParameter = []
                            aParameter = aKineticLaw.getParameter( NumPara )

                            anId_KL_P = aParameter.getId()
                            aName_KL_P = aParameter.getName()
                            aValue_KL_P = str( aParameter.getValue() )
                            aUnit_KL_P = aParameter.getUnits()
                            aConstant_KL_P = aParameter.getConstant()

                            ListOfParameter.append( anId_KL_P )
                            ListOfParameter.append( aName_KL_P )
                            ListOfParameter.append( aValue_KL_P )
                            ListOfParameter.append( aUnit_KL_P )
                            ListOfParameter.append( aConstant_KL_P )

                            ListOfParameters.append( ListOfParameter )
                    else:
                        ListOfParameters = []

                    anExpressionAnnotation = aKineticLaw.getAnnotation()

                    ListOfKineticLaw.append( aFormula_KL )
                    ListOfKineticLaw.append( aString_KL )
                    ListOfKineticLaw.append( aTimeUnit_KL )
                    ListOfKineticLaw.append( aSubstanceUnit_KL )
                    ListOfKineticLaw.append( ListOfParameters )
                    ListOfKineticLaw.append( anExpressionAnnotation )

#---------------------------------------------------------


            aReversible = aReaction.getReversible()
            aFast = aReaction.getFast()


            ListOfReactants = []
            if aReaction.getReactant(0):
                NumReactant = aReaction.getNumReactants()
                for NumR in range( NumReactant ):
                    ListOfReactant = []

                    aSpeciesReference= aReaction.getReactant( NumR )

                    aSpecies_R = aSpeciesReference.getSpecies()
                    aStoichiometry_R = aSpeciesReference.getStoichiometry()

                    aString_R = []
                    if aSpeciesReference.isSetStoichiometryMath():
                        aNode_R = aSpeciesReference.getStoichiometryMath()
                        if aNode_R.isSetMath():
                            aString_R = sub( libsbml.formulaToString,
                                aNode_R.getMath() )

                    aDenominator_R = aSpeciesReference.getDenominator()

                    ListOfReactant.append( aSpecies_R )

                    if aStoichiometry_R == [] and aString_R == []:
                        ListOfReactant.append( aStoichiometry_R )
                    elif aStoichiometry_R != [] and aString_R == []:
                        ListOfReactant.append( aStoichiometry_R )
                    elif aStoichiometry_R == [] and aString_R != []:
                        ListOfReactant.append( aString_R )
                    elif aStoichiometry_R != [] and aString_R != []:
                        ListOfReactant.append( aStoichiometry_R )
                        ListOfReactant.append( aString_R )

                    ListOfReactant.append( aDenominator_R )

                    ListOfReactants.append( ListOfReactant )


            ListOfProducts = []
            if aReaction.getProduct(0):
                NumProduct = aReaction.getNumProducts()
                for NumP in range( NumProduct ):
                    ListOfProduct = []

                    aSpeciesReference = aReaction.getProduct( NumP )

                    aSpecies_P = aSpeciesReference.getSpecies()
                    aStoichiometry_P = aSpeciesReference.getStoichiometry()

                    aString_P = []
                    if aSpeciesReference.isSetStoichiometryMath():
                        aNode_P = aSpeciesReference.getStoichiometryMath()
                        if aNode_P.isSetMath():
                            aString_P = sub( libsbml.formulaToString,
                                    aNode_P.getMath() )

                    aDenominator_P = aSpeciesReference.getDenominator()

                    ListOfProduct.append( aSpecies_P )

                    if aStoichiometry_P == [] and aString_P == []:
                        ListOfProduct.append( aStoichiometry_P )
                    elif aStoichiometry_P != [] and aString_P == []:
                        ListOfProduct.append( aStoichiometry_P )
                    elif aStoichiometry_P == [] and aString_P != []:
                        ListOfProduct.append( aString_P )
                    elif aStoichiometry_P != [] and aString_P != []:
                        ListOfProduct.append( aStoichiometry_P )
                        ListOfProduct.append( aString_P )

                    ListOfProduct.append( aDenominator_P )

                    ListOfProducts.append( ListOfProduct )


            ListOfModifiers = []
            if aReaction.getModifier(0):
                NumModifier = aReaction.getNumModifiers()
                for NumM in range( NumModifier ):
                    aSpeciesReference = aReaction.getModifier( NumM )

                    aSpecies_M = aSpeciesReference.getSpecies()
                    ListOfModifiers.append( aSpecies_M )

            ListOfReaction.append( anId )
            ListOfReaction.append( aName )
            ListOfReaction.append( ListOfKineticLaw )
            ListOfReaction.append( aReversible )
            ListOfReaction.append( aFast )
            ListOfReaction.append( ListOfReactants )
            ListOfReaction.append( ListOfProducts )
            ListOfReaction.append( ListOfModifiers )
            LIST.append( ListOfReaction )

    return LIST


def getRule( aSBMLmodel ):
    " [[ RuleType, Formula, Variable ]] "
    LIST = []
    if aSBMLmodel.getRule(0):
        NumRule = aSBMLmodel.getNumRules()
        for Num in range( NumRule ):
            ListOfRules = []
            aRule = aSBMLmodel.getRule( Num )

            aRuleType = aRule.getTypeCode()
            aFormula = aRule.getFormula()
             
            if ( aRuleType == libsbml.SBML_ALGEBRAIC_RULE ):

                aVariable = ''

            elif ( aRuleType == libsbml.SBML_ASSIGNMENT_RULE or
                   aRuleType == libsbml.SBML_RATE_RULE ):
                
                aVariable = aRule.getVariable()

            elif ( aRuleType == libsbml.SBML_SPECIES_CONCENTRATION_RULE ):

                aVariable = aRule.getSpecies()

            elif( aRuleType == libsbml.SBML_COMPARTMENT_VOLUME_RULE ):
                    
                aVariable = aRule.getCompartment()

            elif( aRuleType == libsbml.SBML_PARAMETER_RULE ):

                aVariable = aRule.getName()
                
            else:
                raise TypeError, " The type of Rule must be Algebraic, Assignment or Rate Rule"

            ListOfRules.append( aRuleType )
            ListOfRules.append( aFormula )
            ListOfRules.append( aVariable )

            LIST.append( ListOfRules )

    return LIST


def getSpecies( aSBMLmodel, DerivedValueDic ):
    " [[ Id , Name , Compartment , InitialAmount , InitialConcentration , SubstanceUnit , SpatialSizeUnit , Unit , HasOnlySubstanceUnit , BoundaryCondition , Charge , Constant ]] "
    LIST = []
    if aSBMLmodel.getSpecies(0):
        NumSpecies = aSBMLmodel.getNumSpecies()
        for Num in range( NumSpecies ):
            ListOfSpecies = []
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
                if getInitialValueFromAssignmentRule( aSBMLmodel, anId_Sp, DerivedValueDic ):
                    anInitialAmount_Sp = DerivedValueDic[ anId_Sp ]

            aSubstanceUnit_Sp = aSpecies.getSubstanceUnits()
            aSpatialSizeUnit_Sp = aSpecies.getSpatialSizeUnits()
            anUnit_Sp = aSpecies.getUnits()
            aHasOnlySubstanceUnit_Sp = aSpecies.getHasOnlySubstanceUnits()
            aBoundaryCondition_Sp = aSpecies.getBoundaryCondition()
            aCharge_Sp = aSpecies.getCharge()
            aConstant_Sp = aSpecies.getConstant()


            ListOfSpecies.append( anId_Sp )
            ListOfSpecies.append( aName_Sp )
            ListOfSpecies.append( aCompartment_Sp )
            ListOfSpecies.append( anInitialAmount_Sp )
            ListOfSpecies.append( anInitialConcentration_Sp )
            ListOfSpecies.append( aSubstanceUnit_Sp )
            ListOfSpecies.append( aSpatialSizeUnit_Sp )
            ListOfSpecies.append( anUnit_Sp )
            ListOfSpecies.append( aHasOnlySubstanceUnit_Sp )
            ListOfSpecies.append( aBoundaryCondition_Sp )
            ListOfSpecies.append( aCharge_Sp )
            ListOfSpecies.append( aConstant_Sp )
       
            LIST.append( ListOfSpecies )

    return LIST


def getUnitDefinition( aSBMLmodel ):
    " [[ Id , Name , [[ Kind , Exponent , Scale , Multiplier , Offset ]] ]] "
    LIST = []
    if aSBMLmodel.getUnitDefinition(0):
        NumUnitDefinition = aSBMLmodel.getNumUnitDefinitions()
        for Num1 in range( NumUnitDefinition ):
            ListOfUnitDefinition = []

            anUnitDefinition = aSBMLmodel.getUnitDefinition( Num1 )

            anId = anUnitDefinition.getId()
            aName = anUnitDefinition.getName()

            ListOfUnits = []
            if anUnitDefinition.getUnit(0):
                NumUnit = anUnitDefinition.getNumUnits()
                for Num2 in range( NumUnit ):
                    ListOfUnit = []
                    anUnit = anUnitDefinition.getUnit( Num2 )

                    anUnitKind = anUnit.getKind()

                    aKind = libsbml.UnitKind_toString( anUnitKind )
                    anExponent = anUnit.getExponent()
                    aScale = anUnit.getScale()
                    aMultiplier = anUnit.getMultiplier()
                    anOffset = anUnit.getOffset()

                    ListOfUnit.append( aKind  )
                    ListOfUnit.append( anExponent )
                    ListOfUnit.append( aScale )
                    ListOfUnit.append( aMultiplier )
                    ListOfUnit.append( anOffset )

                    ListOfUnits.append( ListOfUnit )

            ListOfUnitDefinition.append( anId )
            ListOfUnitDefinition.append( aName )
            ListOfUnitDefinition.append( ListOfUnits )

                
            LIST.append( ListOfUnitDefinition )

    return LIST


def getInitialValueFromAssignmentRule( aSBMLmodel, aVariableID, DerivedValueDic ):
    '''
    return False when the initial value can not calculate.
    '''

    if ( aVariableID in DerivedValueDic ):
        return DerivedValueDic[ aVariableID ]

    anAssignmentRule = aSBMLmodel.getAssignmentRule( aVariableID )
    if ( anAssignmentRule == None ):
        return False                                                   ### FIXME

    aFormulaTree = anAssignmentRule.getMath().deepCopy()

    print "\nAssignmentRule for %s" % aVariableID
    print "  %s\n" % anAssignmentRule.getFormula()
    print "Initial Construction:\n"
    _dumpTreeConstructionOfNode( aFormulaTree )
    print '\n'

    aCounter = 1
    while ( aCounter > 0 ):
        aCounter = 0
        _convertName2Value( aSBMLmodel, aFormulaTree, aCounter, DerivedValueDic )

    print "Name replaced with value:\n"
    _dumpTreeConstructionOfNode( aFormulaTree )
    print '\n'

##    _dumpTreeConstructionOfNode( aFormulaTree )

    while ( aFormulaTree.getNumChildren() > 0 ):
        _calcInitialValue( aFormulaTree )

##    print "Initial Value of %s: %s" % ( aVariableID, _getNodeValue( aFormulaTree ))

    DerivedValueDic[ aVariableID ] = _getNodeValue( aFormulaTree )
    return True


def _convertName2Value( aSBMLmodel, aNode, aCounter, DerivedValueDic ):

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
                if getInitialValueFromAssignmentRule( aSBMLmodel, aVariableID, DerivedValueDic ):
                    aNode.setType( libsbml.AST_REAL )
                    aNode.setValue( DerivedValueDic[ aVariableID ] )
                else:
                    aCounter += 1
        
        elif ( anElementType == libsbml.SBML_PARAMETER ):
            if anElement.isSetValue() == True:
                aNode.setType( libsbml.AST_REAL )
                aNode.setValue( anElement.getValue() )
            else:
                if getInitialValueFromAssignmentRule( aSBMLmodel, aVariableID, DerivedValueDic ):
                    aNode.setType( libsbml.AST_REAL )
                    aNode.setValue( DerivedValueDic[ aVariableID ] )
                else:
                    aCounter += 1
        
        elif ( anElementType == libsbml.SBML_COMPARTMENT ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( anElement.getSize() )
        
        else:
            raise TypeError,\
            "_convertName2Value: Element type must be Species, Parameter, or Compartment"

    for i in range( aNode.getNumChildren() ):
        _convertName2Value( aSBMLmodel, aNode.getChild( i ), aCounter, DerivedValueDic )


def _calcInitialValue( aNode ):
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
            aNode.setValue( float( _getNodeValue( aNode.getLeftChild())) / float( _getNodeValue( aNode.getRightChild())) )

        elif( aNodeType == libsbml.AST_MINUS ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( float( _getNodeValue( aNode.getLeftChild())) - float( _getNodeValue( aNode.getRightChild())) )

        elif( aNodeType == libsbml.AST_PLUS ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( float( _getNodeValue( aNode.getLeftChild())) + float( _getNodeValue( aNode.getRightChild())) )

        elif ( aNodeType == libsbml.AST_TIMES ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( float( _getNodeValue( aNode.getLeftChild())) * float( _getNodeValue( aNode.getRightChild())) )

        elif ( aNodeType == libsbml.AST_POWER ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( pow( float( _getNodeValue( aNode.getLeftChild())), float( _getNodeValue( aNode.getRightChild()))) )

        ## Logical operators

        elif ( aNodeType == libsbml.AST_LOGICAL_AND ):
            aNode.setType( libsbml.AST_INTEGER )
            aNode.setValue( _getNodeValue( aNode.getLeftChild() ) and _getNodeValue( aNode.getRightChild() ) )

        elif ( aNodeType == libsbml.AST_LOGICAL_NOT ):
            aNode.setType( libsbml.AST_INTEGER )
            aNode.setValue( not _getNodeValue( aNode.getChild( 0 )))

        elif ( aNodeType == libsbml.AST_LOGICAL_OR ):
            aNode.setType( libsbml.AST_INTEGER )
            aNode.setValue( _getNodeValue( aNode.getLeftChild() ) or _getNodeValue( aNode.getRightChild() ) )

        elif ( aNodeType == libsbml.AST_LOGICAL_XOR ):
            aNode.setType( libsbml.AST_INTEGER )
            aNode.setValue( ( not( _getNodeValue( aNode.getLeftChild() ))) and ( not( _getNodeValue( aNode.getRightChild() ))))

        ## Relational operators

        elif ( aNodeType == libsbml.AST_RELATIONAL_EQ ):
            aNode.setType( libsbml.AST_INTEGER )
            if ( float( _getNodeValue( aNode.getLeftChild() )) == float( _getNodeValue( aNode.getRightChild() )) ):
                aNode.setValue( 1 )
            else:
                aNode.setValue( 0 )

        elif ( aNodeType == libsbml.AST_RELATIONAL_GEQ ):
            aNode.setType( libsbml.AST_INTEGER )
            if ( float( _getNodeValue( aNode.getLeftChild() )) >= float( _getNodeValue( aNode.getRightChild() )) ):
                aNode.setValue( 1 )
            else:
                aNode.setValue( 0 )

        elif ( aNodeType == libsbml.AST_RELATIONAL_GT ):
            aNode.setType( libsbml.AST_INTEGER )
            if ( float( _getNodeValue( aNode.getLeftChild() )) > float( _getNodeValue( aNode.getRightChild() )) ):
                aNode.setValue( 1 )
            else:
                aNode.setValue( 0 )

        elif ( aNodeType == libsbml.AST_RELATIONAL_LEQ ):
            aNode.setType( libsbml.AST_INTEGER )
            if ( float( _getNodeValue( aNode.getLeftChild() )) <= float( _getNodeValue( aNode.getRightChild() )) ):
                aNode.setValue( 1 )
            else:
                aNode.setValue( 0 )

        elif ( aNodeType == libsbml.AST_RELATIONAL_LT ):
            aNode.setType( libsbml.AST_INTEGER )
            if ( float( _getNodeValue( aNode.getLeftChild() )) < float( _getNodeValue( aNode.getRightChild() )) ):
                aNode.setValue( 1 )
            else:
                aNode.setValue( 0 )

        elif ( aNodeType == libsbml.AST_RELATIONAL_NEQ ):
            aNode.setType( libsbml.AST_INTEGER )
            if ( float( _getNodeValue( aNode.getLeftChild() )) != float( _getNodeValue( aNode.getRightChild() )) ):
                aNode.setValue( 1 )
            else:
                aNode.setValue( 0 )

        ## Functions

        ##     Power and logarithmic functions

        elif ( aNodeType == libsbml.AST_FUNCTION_EXP ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.exp( float( _getNodeValue( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_LN ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.log( float( _getNodeValue( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_LOG ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.log( float( _getNodeValue( aNode.getLeftChild())), float( _getNodeValue( aNode.getRightChild() ))))

        elif ( aNodeType == libsbml.AST_FUNCTION_POWER ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( pow( float( _getNodeValue( aNode.getLeftChild())), float( _getNodeValue( aNode.getRightChild() ))))

        elif ( aNodeType == libsbml.AST_FUNCTION_ROOT ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( pow( float( _getNodeValue( aNode.getLeftChild())), 1.0 / float( _getNodeValue( aNode.getRightChild() ))))

        ##     Number-theoretic and representation functions

        elif ( aNodeType == libsbml.AST_FUNCTION_ABS ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( abs( float( _getNodeValue( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_CEILING ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.ceil( float( _getNodeValue( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_FACTORIAL ):
            aNode.setType( libsbml.AST_INTEGER )
            aNode.setValue( math.factorial( _getNodeValue( aNode.getChild( 0 ))))

        elif ( aNodeType == libsbml.AST_FUNCTION_FLOOR ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.floor( float( _getNodeValue( aNode.getChild( 0 )))))

        ##     Trigonometric functions

        elif ( aNodeType == libsbml.AST_FUNCTION_SIN ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.sin( float( _getNodeValue( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_COS ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.cos( float( _getNodeValue( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_TAN ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.tan( float( _getNodeValue( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_CSC ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( 1.0 / math.sin( float( _getNodeValue( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_SEC ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( 1.0 / math.cos( float( _getNodeValue( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_COT ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( 1.0 / math.tan( float( _getNodeValue( aNode.getChild( 0 )))))

        ##     Inverse trigonometric functions

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCSIN ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.asin( float( _getNodeValue( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCCOS ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.acos( float( _getNodeValue( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCTAN ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.atan( float( _getNodeValue( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCCSC ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( 1.0 / math.asin( float( _getNodeValue( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCSEC ):  ## arc-secant
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.acos( 1.0 / float( _getNodeValue( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCCOT ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.atan( 1.0 / float( _getNodeValue( aNode.getChild( 0 )))))

        ##     Hyperbolic functions

        elif ( aNodeType == libsbml.AST_FUNCTION_SINH ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.sinh( float( _getNodeValue( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_COSH ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.cosh( float( _getNodeValue( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_TANH ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( math.tanh( float( _getNodeValue( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_CSCH ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( 1.0 / math.sinh( float( _getNodeValue( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_SECH ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( 1.0 / math.cosh( float( _getNodeValue( aNode.getChild( 0 )))))

        elif ( aNodeType == libsbml.AST_FUNCTION_COTH ):
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( 1.0 / math.tanh( float( _getNodeValue( aNode.getChild( 0 )))))

        ##     Inverse hyperbolic functions

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCSINH ):
            aNode.setType( libsbml.AST_REAL )
            if geqPython26:
                aNode.setValue( math.asinh( float( _getNodeValue( aNode.getChild( 0 )))))
            else:
                x = float( _getNodeValue( aNode.getChild( 0 )))
                aNode.setValue( math.log( x + math.sqrt( x * x + 1.0 )))

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCCOSH ):
            aNode.setType( libsbml.AST_REAL )
            if geqPython26:
                aNode.setValue( math.acosh( float( _getNodeValue( aNode.getChild( 0 )))))
            else:
                x = float( _getNodeValue( aNode.getChild( 0 )))
                aNode.setValue( math.log( x + math.sqrt( x + 1.0 ) * math.sqrt( x - 1.0 )))

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCTANH ):
            aNode.setType( libsbml.AST_REAL )
            if geqPython26:
                aNode.setValue( math.atanh( float( _getNodeValue( aNode.getChild( 0 )))))
            else:
                x = float( _getNodeValue( aNode.getChild( 0 )))
                aNode.setValue( 0.5 * math.log( ( 1.0 + x ) / ( 1.0 - x )))

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCCSCH ):
            aNode.setType( libsbml.AST_REAL )
            if geqPython26:
                aNode.setValue( math.asinh( 1.0 / float( _getNodeValue( aNode.getChild( 0 )))))
            else:
                x = float( _getNodeValue( aNode.getChild( 0 )))
                aNode.setValue( math.log( 1.0 / x + math.sqrt( 1.0 / x / x + 1.0 )))

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCSECH ):
            aNode.setType( libsbml.AST_REAL )
            if geqPython26:
                aNode.setValue( math.acosh( 1.0 / float( _getNodeValue( aNode.getChild( 0 )))))
            else:
                x = float( _getNodeValue( aNode.getChild( 0 )))
                aNode.setValue( math.log( 1.0 / x + math.sqrt( 1.0 / x + 1.0 ) * math.sqrt( 1.0 / x - 1.0 )))

        elif ( aNodeType == libsbml.AST_FUNCTION_ARCCOTH ):
            aNode.setType( libsbml.AST_REAL )
            if geqPython26:
                aNode.setValue( math.atanh( 1.0 / float( _getNodeValue( aNode.getChild( 0 )))))
            else:
                x = float( _getNodeValue( aNode.getChild( 0 )))
                aNode.setValue( 0.5 * math.log( ( x + 1.0 ) / ( x - 1.0 )))

        ##     Other functions

        elif ( aNodeType == libsbml.AST_FUNCTION_DELAY ):
            ### At t = 0, delayed value is not available, thus value at t = 0 is used.
            aNode.setType( libsbml.AST_REAL )
            aNode.setValue( _getNodeValue( aNode.getChild( 0 )))

        elif ( aNodeType == libsbml.AST_FUNCTION_PIECEWISE ):
##            print "\nCulc AST_FUNCTION_PIECEWISE:"
##            _dumpTreeConstructionOfNode( aNode )
##
##            print "  #Children = %d" % aNode.getNumChildren()
            for i in range( aNode.getNumChildren() / 2 ):
                if ( _getNodeValue( aNode.getChild( i * 2 + 1 )) != 0 ):
##                    aNode = aNode.getChild( i * 2 ).deepCopy()
                    aNode.setType( aNode.getChild( i * 2 ).getType() )
                    aNode.setValue( _getNodeValue( aNode.getChild( i * 2 )))
##                    print "  piece(%d) is True! Value is %s" % ( i, _getNodeValue( aNode ))
                    break
            
            if ( aNode.getType() == libsbml.AST_FUNCTION_PIECEWISE ):
                if ( aNode.getNumChildren() % 2 == 1 ):
##                    aNode = aNode.getRightChild().deepCopy()
                    aNode.setType( aNode.getRightChild().getType() )
                    aNode.setValue( _getNodeValue( aNode.getRightChild()))
                else:
                    raise TypeError,\
                    "Can't derive an initial value from a piecewise function"

        ## Unknown

        elif ( aNodeType == libsbml.AST_UNKNOWN ):
            raise TypeError,\
            "Unknown operator is detected in Formula"

        '''
AST_FUNCTION            : Solved by converter
AST_LAMBDA
AST_NAME_TIME
        '''

        _removeAllChildren( aNode )

    for i in range( aNode.getNumChildren() ):
        _calcInitialValue( aNode.getChild( i ) )


def _getNodeValue( aNode ):
    aNodeType = aNode.getType()
    if ( aNode.isReal() ):
        return aNode.getReal()
    elif ( aNode.isInteger() ):
        return aNode.getInteger()
    else:
        raise TypeError,\
        "aNode must be number"


def _removeAllChildren( aNode ):
    while ( aNode.getNumChildren() > 0 ):
        aNode.removeChild( 0 )
    return


def _dumpTreeConstructionOfNode( aNode ):

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
        _dumpTreeConstructionOfNode( aNode.getChild( i ) )


