import libsbml

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
        
    NumCompartment = theList.getNumItems()
        
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
            
            anASTNode_Ev_Tr = anEvent.getTrigger()
            aString_Ev_Tr = sub( libsbml.formulaToString , anASTNode_Ev_Tr )
            
            anASTNode_Ev_De = anEvent.getDelay()
            aString_Ev_De = sub( libsbml.formulaToString , anASTNode_Ev_Tr )
            
            aTimeUnit_Ev = anEvent.getTimeUnits()
            
            ListOfEventAssignments = []
            if anEvent.getEventAssignment(0):
                NumEventAssignment = anEvent.getNumEventAssignments()
                for Num_Ev_As in range( NumEventAssignment ):
                    ListOfEventAssignment = []
                    
                    anEventAssignment = anEvent.getEventAssignment( Num_Ev_As )
                    
                    aVariable_Ev_As = anEventAssignment.getVariable()
                    
                    anASTNode_Ev_As = anEventAssignment.getMath()
                    aString_Ev_As = sub( libsbml.formulaToString , anASTNode_Ev_As )
                    
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

            anASTNode_FD = aFunctionDefinition.getMath()
            aString_FD = sub( libsbml.formulaToString , anASTNode_FD )

            ListOfFunctionDefinition.append( anId_FD )
            ListOfFunctionDefinition.append( aName_FD )
            ListOfFunctionDefinition.append( aString_FD )

            LIST.append( ListOfFunctionDefinition )

    return LIST


def getParameter( aSBMLmodel ):
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
                'Unknown'
                
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
            if aReaction.isSetKineticLaw():

                aKineticLaw = aReaction.getKineticLaw()
                #            anASTNode = libsbml.ASTNode()
                ListOfKineticLaw = []
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
                            anASTNode_KL = aKineticLaw.getMath()
                            aMath.append( libsbml.formulaToString\
                                          ( anASTNode_KL ) )
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

                    if aSpeciesReference.isSetStoichiometryMath():
                        anASTNode_R = aSpeciesReference.getStoichiometryMath()
                        aString_R = sub( libsbml.formulaToString , anASTNode_R )
                    else:
                        aString_R = []

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

                    if aSpeciesReference.isSetStoichiometryMath():
                        anASTNode_P = aSpeciesReference.getStoichiometryMath()
                        aString_P = sub( libsbml.formulaToString , anASTNode_P )
                    else:
                        aString_P = []

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


def getSpecies( aSBMLmodel ):
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

