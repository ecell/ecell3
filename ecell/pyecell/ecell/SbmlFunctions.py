from sbml import *

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
    if Model_getCompartment( aSBMLmodel , 0 ):
        NumCompartment = Model_getNumCompartments( aSBMLmodel )
        for Num in range( NumCompartment ):
            ListOfCompartment = []
            aCompartment = Model_getCompartment( aSBMLmodel , Num )
            
            anId = sub( Compartment_getId , aCompartment )
            aName = sub( Compartment_getName , aCompartment )
            aSpatialDimension = Compartment_getSpatialDimensions( aCompartment )
            if Compartment_isSetSize( aCompartment ):
                aSize = sub( Compartment_getSize , aCompartment )
            else:
                aSize = "Unknown"

            if Compartment_isSetVolume( aCompartment ):
                aVolume = sub( Compartment_getVolume , aCompartment )
            else:
                aVoluem = "Unknown"
                
            anUnit = sub( Compartment_getUnits , aCompartment )
            anOutside = sub( Compartment_getOutside , aCompartment )
            aConstant = Compartment_getConstant( aCompartment )

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
    if Model_getEvent( aSBMLmodel , 0 ):
        NumEvent = Model_getNumEvents( aSBMLmodel )
        for Num_Ev in range( NumEvent ):
            ListOfEvent = []
            
            anEvent = Model_getEvent( aSBMLmodel , Num_Ev )
            
            anId_Ev = sub( Event_getId , anEvent )
            aName_Ev = sub( Event_getName , anEvent )
            
            anASTNode_Ev_Tr = sub( Event_getTrigger , anEvent )
            aString_Ev_Tr = sub( SBML_formulaToString , anASTNode_Ev_Tr )
            
            anASTNode_Ev_De = sub( Event_getDelay , anEvent )
            aString_Ev_De = sub( SBML_formulaToString , anASTNode_Ev_Tr )
            
            aTimeUnit_Ev = sub( Event_getTimeUnits , anEvent )
            
            ListOfEventAssignments = []
            if Event_getEventAssignment( anEvent , 0 ):
                NumEventAssignment = Event_getNumEventAssignments( anEvent )
                for Num_Ev_Ass in range( NumEventAssignment ):
                    ListOfEventAssignment = []
                    
                    anEventAssignment = Event_getEventAssignment( anEvent , Num_Ev_Ass )
                    
                    aVariable_Ev_Ass = sub( EventAssignment_getVariable , anEventAssignment )
                    
                    anASTNode_Ev_Ass = sub( EventAssignment_getMath , anEventAssignment )
                    aString_Ev_Ass = sub( SBML_formulaToString , anASTNode_Ev_Ass )
                    
                    ListOfEventAssignment.append( aVariable_Ev_Ass )
                    ListOfEventAssignment.append( aString_Ev_Ass )
                    
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
    if Model_getFunctionDefinition( aSBMLmodel , 0 ):
        NumFunctionDefinition = Model_getNumFunctionDefinitions( aSBMLmodel )
        for Num_FD in range( NumFunctionDefinition ):
            ListOfFunctionDefinition = []

            aFunctionDefinition = Model_getFunctionDefinition( aSBMLmodel , Num_FD )

            anId_FD = sub( FunctionDefinition_getId , aFunctionDefinition )
            aName_FD = sub( FunctionDefinition_getName , aFunctionDefinition )

            anASTNode_FD = sub( FunctionDefinition_getMath , aFunctionDefinition )
            aString_FD = sub( SBML_formulaToString , anASTNode_FD )

            ListOfFunctionDefinition.append( anId_FD )
            ListOfFunctionDefinition.append( aName_FD )
            ListOfFunctionDefinition.append( aString_FD )

            LIST.append( ListOfFunctionDefinition )
    return LIST


def getParameter( aSBMLmodel ):
    " [[ Id , Name , Value , Unit , Constant ]] "
    LIST = []
    if Model_getParameter( aSBMLmodel , 0 ):
        NumParameter = Model_getNumParameters( aSBMLmodel )
        for Num_Pa in range( NumParameter ):
            ListOfParameter = []

            aParameter = Model_getParameter( aSBMLmodel , Num_Pa )

            anId_Pa = sub( Parameter_getId , aParameter )
            aName_Pa = sub( Parameter_getName , aParameter )
            
            if Parameter_isSetValue( aParameter ):
                aValue_Pa = Parameter_getValue( aParameter )
            else:
                "Unknown"
                
            anUnit_Pa = sub( Parameter_getUnits  , aParameter )
            aConstant_Pa = sub( Parameter_getConstant , aParameter )

            ListOfParameter.append( anId_Pa )
            ListOfParameter.append( aName_Pa )
            ListOfParameter.append( aValue_Pa )
            ListOfParameter.append( anUnit_Pa )
            ListOfParameter.append( aConstant_Pa )

            LIST.append( ListOfParameter )

    return LIST


def getReaction( aSBMLmodel ):
    " [[ Id , Name , [ Formula , String , TimeUnit , SubstanceUnit , [[ ParameterId , ParameterName , ParameterValue , ParameterUnit , ParameterConstant ]] ] , Reversible , Fast , [[ ReactantSpecies , ( ReactantStoichiometry , ReactantStoichiometryMath ) , ReactantDenominator  ]] , [[  ProductSpecies , ( ProductStoichiometry , ProductStoichiometryMath ) , ProductDenominator ]] , [[ ModifierSpecies ]] ]] "
    LIST = []
    if Model_getReaction( aSBMLmodel , 0 ):
        NumReaction = Model_getNumReactions( aSBMLmodel )
        for Num in range( NumReaction ):
            ListOfReaction = []
            aReaction = Model_getReaction( aSBMLmodel , Num )

            anId = sub( Reaction_getId , aReaction )
            aName = sub( Reaction_getName , aReaction )

#----------KineticLaw----------------------------------
            aKineticLaw = sub( Reaction_getKineticLaw , aReaction )
            ListOfKineticLaw = []
            if aKineticLaw != []:
            
                aFormula_KL = sub( KineticLaw_getFormula , aKineticLaw )

                anASTNode_KL = sub( KineticLaw_getMath , aKineticLaw )
                aString_KL = sub( SBML_formulaToString , anASTNode_KL )

                aTimeUnit_KL = sub( KineticLaw_getTimeUnits , aKineticLaw )
                aSubstanceUnit_KL = sub( KineticLaw_getSubstanceUnits , aKineticLaw )
            
                if KineticLaw_getParameter( aKineticLaw , 0 ) :
                    ListOfParameters = []
                    NumParameter_KL = KineticLaw_getNumParameters( aKineticLaw )
                    for NumPara in range( NumParameter_KL ):
                        ListOfParameter = []
                        aParameter_KL = KineticLaw_getParameter( aKineticLaw , NumPara )

                        anId_KL_P = sub( Parameter_getId , aParameter_KL )
                        aName_KL_P = sub( Parameter_getName , aParameter_KL )
                        aValue_KL_P = str( Parameter_getValue( aParameter_KL ) )
                        aUnit_KL_P = sub( Parameter_getUnits , aParameter_KL )
                        aConstant_KL_P = sub( Parameter_getConstant , aParameter_KL )

                        ListOfParameter.append( anId_KL_P )
                        ListOfParameter.append( aName_KL_P )
                        ListOfParameter.append( aValue_KL_P )
                        ListOfParameter.append( aUnit_KL_P )
                        ListOfParameter.append( aConstant_KL_P )

                        ListOfParameters.append( ListOfParameter )
                else:
                    ListOfParameters = []

                ListOfKineticLaw.append( aFormula_KL )
                ListOfKineticLaw.append( aString_KL )
                ListOfKineticLaw.append( aTimeUnit_KL )
                ListOfKineticLaw.append( aSubstanceUnit_KL )
                ListOfKineticLaw.append( ListOfParameters )

#---------------------------------------------------------


            aReversible = sub( Reaction_getReversible , aReaction )
            aFast = sub( Reaction_getFast , aReaction )


            ListOfReactants = []
            if Reaction_getReactant( aReaction , 0 ):
                NumReactant = Reaction_getNumReactants( aReaction )
                for NumR in range( NumReactant ):
                    ListOfReactant = []

                    aSpeciesReference_Reactant = Reaction_getReactant( aReaction , NumR )

                    aSpecies_R = sub( SpeciesReference_getSpecies , aSpeciesReference_Reactant )
                    aStoichiometry_R = sub( SpeciesReference_getStoichiometry , aSpeciesReference_Reactant )

                    anASTNode_R = sub( SpeciesReference_getStoichiometryMath , aSpeciesReference_Reactant )
                    aString_R = sub( SBML_formulaToString , anASTNode_R )

                    aDenominator_R = sub( SpeciesReference_getDenominator , aSpeciesReference_Reactant )

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
            if Reaction_getProduct( aReaction , 0 ):
                NumProduct = Reaction_getNumProducts( aReaction )
                for NumP in range( NumProduct ):
                    ListOfProduct = []

                    aSpeciesReference_Product = Reaction_getProduct( aReaction , NumP )

                    aSpecies_P = sub( SpeciesReference_getSpecies , aSpeciesReference_Product )
                    aStoichiometry_P = sub( SpeciesReference_getStoichiometry , aSpeciesReference_Product )
                    
                    anASTNode_P = sub( SpeciesReference_getStoichiometryMath , aSpeciesReference_Product )
                    aString_P = sub( SBML_formulaToString , anASTNode_P )

                    aDenominator_P = sub( SpeciesReference_getDenominator , aSpeciesReference_Product )

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
            if Reaction_getModifier( aReaction , 0 ):
                NumModifier = Reaction_getNumModifiers( aReaction )
                for NumM in range( NumModifier ):
                    aModifierSpeciesReference = Reaction_getModifier( aReaction , NumM )

                    aSpecies_M = sub( ModifierSpeciesReference_getSpecies , aModifierSpeciesReference )
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
    " [[ Formula ]] "
    LIST = []
    if Model_getRule( aSBMLmodel , 0 ):
        NumRule = Model_getNumRules( aSBMLmodel )
        for Num in range( NumRule ):
            aRule = Model_getRule( aSBMLmodel , Num )
            #'Model_getAssignmentRule() was not found   
            aFormula = sub( Rule_getFormula , aRule )

            LIST.append( aFormula )

    return LIST


def getSpecies( aSBMLmodel ):
    " [[ Id , Name , Compartment , InitialAmount , InitialConcentration , SubstanceUnit , SpatialSizeUnit , Unit , HasOnlySubstanceUnit , BoundaryCondition , Charge , Constant ]] "
    LIST = []
    if Model_getSpecies( aSBMLmodel , 0 ):
        NumSpecies = Model_getNumSpecies( aSBMLmodel )
        for Num in range( NumSpecies ):
            ListOfSpecies = []
            aSpecies = Model_getSpecies( aSBMLmodel , Num )

            anId_Sp = sub( Species_getId , aSpecies )
            aName_Sp = sub( Species_getName ,aSpecies )
            aCompartment_Sp = sub( Species_getCompartment, aSpecies )

            if Species_isSetInitialAmount( aSpecies ):
                anInitialAmount_Sp = Species_getInitialAmount( aSpecies )
            else:
                anInitialAmount_Sp = "Unknown"

            if Species_isSetInitialConcentration( aSpecies ):
                anInitialConcentration_Sp = Species_getInitialConcentration( aSpecies )
            else:
                anInitialConcentration_Sp = "Unknown"
                
            aSubstanceUnit_Sp = sub( Species_getSubstanceUnits , aSpecies )
            aSpatialSizeUnit_Sp = sub( Species_getSpatialSizeUnits , aSpecies )
            anUnit_Sp = sub( Species_getUnits , aSpecies )
            aHasOnlySubstanceUnit_Sp = Species_getHasOnlySubstanceUnits( aSpecies )
            aBoundaryCondition_Sp = Species_getBoundaryCondition( aSpecies )
            aCharge_Sp = sub( Species_getCharge , aSpecies )
            aConstant_Sp = Species_getConstant( aSpecies )


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
    if Model_getUnitDefinition( aSBMLmodel , 0 ):
        NumUnitDefinition = Model_getNumUnitDefinitions( aSBMLmodel )
        for Num1 in range( NumUnitDefinition ):
            ListOfUnitDefinition = []

            anUnitDefinition = Model_getUnitDefinition( aSBMLmodel , Num1 )

            anId = sub( UnitDefinition_getId , anUnitDefinition )
            aName = sub( UnitDefinition_getName , anUnitDefinition )

            ListOfUnits = []
            if UnitDefinition_getUnit( anUnitDefinition , 0 ):
                NumUnit = UnitDefinition_getNumUnits( anUnitDefinition )
                for Num2 in range( NumUnit ):
                    ListOfUnit = []
                    anUnit = UnitDefinition_getUnit( anUnitDefinition , Num2 )

                    anUnitKind = sub( Unit_getKind , anUnit )

                    aKind = sub( UnitKind_toString , anUnitKind )
                    anExponent = sub( Unit_getExponent , anUnit )
                    aScale = Unit_getScale( anUnit )
                    aMultiplier = sub( Unit_getMultiplier , anUnit )
                    anOffset = Unit_getOffset( anUnit )

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


#if __name__ == '__main__':
#    import sys

#    filename=sys.argv[1]
#    aSBMLdocument = readSBML( filename )
#    aSBMLmodel = SBMLDocument_getModel( aSBMLdocument )

#    aCompartment = getCompartment( aSBMLmodel )
#    anEvent = getEvent( aSBMLmodel )
#    aFunctionDefinition = getFunctionDefinition( aSBMLmodel )
#    aParameter = getParameter( aSBMLmodel )
#    aReaction = getReaction( aSBMLmodel )
#    aRule = getRule( aSBMLmodel )
#    aSpecies = getSpecies( aSBMLmodel )
#    aUnitDefinition = getUnitDefinition( aSBMLmodel )

#    print aCompartment,'\n\n',\
#          anEvent,'\n\n',\
#          aFunctionDefinition,'\n\n',\
#          aParameter,'\n\n' , \
#          aReaction,'\n\n',\
#          aRule,'\n\n',\
#          aSpecies,'\n\n',\
#          aUnitDefinition,'\n\n'
