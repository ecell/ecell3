import libsbml
from SbmlFunctions import *

import sys
import string

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
    

# Avogadro Number set
N_A = 6.0221367e+23





# --------------------------------
# Model Class
# --------------------------------

class SBML_Model:


    def __init__( self, aSBMLDocument, aSBMLmodel ):

        self.CompartmentSize = {}
        self.CompartmentUnit = {}
        self.FunctionDefinition = {}

        self.Level = aSBMLDocument.getLevel()
        self.Version = aSBMLDocument.getVersion() 

        self.CompartmentList = getCompartment( aSBMLmodel )
        self.EventList = getEvent( aSBMLmodel )
        self.FunctionDefinitionList = getFunctionDefinition( aSBMLmodel )
        self.ParameterList = getParameter( aSBMLmodel )
        self.ReactionList = getReaction( aSBMLmodel, aSBMLDocument )
        self.RuleList = getRule( aSBMLmodel )
        self.SpeciesList = getSpecies( aSBMLmodel )
        self.UnitDefinitionList = getUnitDefinition( aSBMLmodel )

        self.setFunctionDefinitionToDictionaly()
        

    # =========================================================

    def setFunctionDefinitionToDictionaly( self ):

        if ( self.FunctionDefinitionList != [] ):

            for aFunctionDefinition in ( self.FunctionDefinitionList ):
                
                self.FunctionDefinition[ aFunctionDefinition[0] ] = aFunctionDefinition[2]
            

    # =========================================================

    #def macroExpand( self, anASTNode ):

    #    aNumChildren = anASTNode.getNumChildren()

    #if ( aNumChildren == 2 ):

    #        aString = self.FunctionDefinition[ anASTNode.getName() ]

    #        anASTNode.getLeftChild().getName()
    #        anASTNode.getRightChild().getName

            
    #    if ( aNumChildren == 1 ):



    # =========================================================

    def getPath( self, aCompartmentID ):

        if( aCompartmentID == 'default' ):
            return '/'
        
        if ( self.Level == 1 ):
            for aCompartment in self.CompartmentList:
                if ( aCompartment[1] == aCompartmentID ):
                    if ( aCompartment[6] == '' or\
                         aCompartment[6] == 'default' ):
                        aPath = '/' + aCompartmentID
                        return aPath
                    else:
                        aPath = self.getPath( aCompartment[6] ) + '/' +\
                                aCompartmentID
                        return aPath

        elif( self.Level == 2 ):
            for aCompartment in self.CompartmentList:
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
            raise Exception, "Version",self.Level," ????"


    # =========================================================

    def getSpeciesReferenceID( self, aSpeciesID ):

        if ( self.Level == 1 ):
            for aSpecies in self.SpeciesList:
                if ( aSpecies[1] == aSpeciesID ):
                    return self.getPath( aSpecies[2] ) + ":" + aSpeciesID
                    
        elif ( self.Level == 2 ):
            for aSpecies in self.SpeciesList:
                if ( aSpecies[0] == aSpeciesID ):
                    return self.getPath( aSpecies[2] ) + ":" + aSpeciesID

        else:
            raise Exception, "Version",self.Level," ????"


    # =========================================================

    def convertUnit( self, aValueUnit, aValue ):

        newValue = []
        if ( self.Level == 1 ):
            for unitList in self.UnitDefinitionList:
                if ( unitList[1] == aValueUnit ):

                    for anUnit in unitList[2]:
                        aValue = aValue * self.__getNewUnitValue( anUnit )

                newValue.append( aValue )

        elif ( self.Level == 2 ):
            for unitList in self.UnitDefinitionList:
                if ( unitList[0] == aValueUnit ):

                    for anUnit in unitList[2]:

                        aValue = aValue * self.__getNewUnitValue( anUnit )

                newValue.append( aValue )

        if( newValue == [] ):
            return aValue
        else:
            return newValue[0]


    # =========================================================

    def __getNewUnitValue( self, anUnit ):

        aValue = 1

        # Scale
        if ( anUnit[2] != 0 ):
            aValue = aValue * pow( 10, anUnit[2] )

        # Multiplier
        aValue = aValue * anUnit[3]

        # Exponent
        aValue = pow( aValue, anUnit[1] )

        # Offset
        aValue = aValue + anUnit[4]

        return aValue

    # =========================================================




# --------------------------------
# Compartment Class
# --------------------------------

class SBML_Compartment( SBML_Model ):


    def __init__( self, aModel ):
        self.Model = aModel

    # =========================================================

    def initialize( self, aCompartment ):

        self.__setSizeToDictionary( aCompartment )
        self.__setUnitToDictionary( aCompartment )

        
    # =========================================================
    
    def getCompartmentID( self, aCompartment ):
        
        if ( aCompartment[6] == '' ):
            if ( self.Model.Level == 1 ):
                aSystemID = '/:' + aCompartment[1]
            elif ( self.Model.Level == 2 ):
                aSystemID = '/:' + aCompartment[0]
            else:
                raise NameError,"Compartment Class needs a ['ID']"

        else:
            if( self.Model.Level == 1 ):
                aSystemID = self.Model.getPath( aCompartment[6] ) + ':'+ aCompartment[1]
            elif( self.Model.Level == 2 ):
                aSystemID = self.Model.getPath( aCompartment[6] ) + ':'+ aCompartment[0]

        return 'System:' + aSystemID


    # =========================================================
    
    def __setSizeToDictionary( self, aCompartment ):

        if( self.Model.Level == 1 ):
            if( aCompartment[4] != "Unknown" ):
                self.Model.CompartmentSize[ aCompartment[1] ] = aCompartment[4]

            else:
                self.Model.CompartmentSize[ aCompartment[1] ] = self.__getOutsideSize( aCompartment[6] )
                
        elif( self.Model.Level == 2 ):
            if( aCompartment[3] != "Unknown" ):
                self.Model.CompartmentSize[ aCompartment[0] ] = aCompartment[3]

            else:
                self.Model.CompartmentSize[ aCompartment[0] ] = self.__getOutsideSize( aCompartment[6] )


    # =========================================================
    
    def __setUnitToDictionary( self, aCompartment ):

        if( self.Model.Level == 1 ):
            aCompartmentID = aCompartment[1]

        elif( self.Model.Level == 2 ):
            aCompartmentID = aCompartment[0]


        if( aCompartment[5] != '' ):
            self.Model.CompartmentUnit[ aCompartmentID ] = aCompartment[5]

        else:
            self.Model.CompartmentUnit[ aCompartmentID ] = self.__getOutsideUnit( aCompartment[6] )


    # =========================================================
    
    def __getOutsideSize( self, anOutsideCompartment ):
        
        if ( anOutsideCompartment == '' ):

            return float( 1 )

        else:
            return self.Model.CompartmentSize[ anOutsideCompartment ]


    # =========================================================
    
    def __getOutsideUnit( self, anOutsideCompartment ):

        if ( anOutsideCompartment == '' ):

            return ''

        else:
            return self.Model.CompartmentUnit[ anOutsideCompartment ]

    # =========================================================    

    def getCompartmentSize( self, aCompartment ):

        if ( self.Model.Level == 1 ):

            return self.Model.CompartmentSize[ aCompartment[1] ]

        elif ( self.Model.Level == 2 ):

            return self.Model.CompartmentSize[ aCompartment[0] ]


    # =========================================================    

    def getCompartmentUnit( self, aCompartment ):

        if ( self.Model.Level == 1 ):

            return self.Model.CompartmentUnit[ aCompartment[1] ]

        elif ( self.Model.Level == 2 ):

            return self.Model.CompartmentUnit[ aCompartment[0] ]


    # =========================================================    




# --------------------------------
# Species Class
# --------------------------------

class SBML_Species( SBML_Model ):

    
    def __init__( self, aModel ):
        self.Model = aModel
    

    # =========================================================
    
    def getSpeciesID( self, aSpecies ):

        aCompartmentID = aSpecies[2]

        if ( aCompartmentID == '' ):
            raise NameError, 'compartment property of Species must be defined'

        if ( self.Model.Level == 1 ):
            aSystemID = self.Model.getPath( aCompartmentID ) + ':' + aSpecies[1]

        elif ( self.Model.Level == 2 ):
            aSystemID = self.Model.getPath( aCompartmentID ) + ':' + aSpecies[0]
        else:
            raise Exception,"Version",self.Level," ????"
                
        return 'Variable:' + aSystemID


    # =========================================================
    
    def getSpeciesValue( self, aSpecies ):

        if ( self.Model.Level == 1 ):

            if ( aSpecies[3] != "Unknown" ):

                if ( aSpecies[7] == '' or aSpecies[7] == 'mole' ):

                    return float( aSpecies[3] ) * N_A

                elif ( aSpecies[7] == 'item' ):

                    return float( aSpecies[3] )

                else:
                    
                    return ( self.Model.convertUnit( aSpecies[7], aSpecies[3] ) ) * N_A

            else:
                raise ValueError, "InitialAmount must be defined, but this model is undefined."


        elif ( self.Model.Level == 2 ):

            if ( aSpecies[3] != "Unknown" ):

                if ( aSpecies[5] != '' and aSpecies[5] != 'mole' ):

                    return ( self.Model.convertUnit( aSpecies[5], aSpecies[3] ) ) * N_A

                elif( aSpecies[5] == 'item' ):

                    return float( aSpecies[3] )

                else:
                    return float( aSpecies[3] ) * N_A
                

            # aSpecies[4] : InitialConcentration
            elif ( aSpecies[4] != "Unknown" ):

                aValue = aSpecies[4]
                aSize = self.Model.CompartmentSize[aSpecies[2]]

                # convert InitialConcentration into molecules number for E-Cell unit
                aValue = aValue * aSize * N_A

                
                # aSpecies[5] : SubstanceUnit
                if ( aSpecies[5] != '' ):

                    # convert Value for matching SubstanceUnit
                    aValue = self.Model.convertUnit( aSpecies[5], aValue )


                # aSpecies[8] : hasOnlySubstanceUnit
                if ( aSpecies[6] != '' ):

                    # convert Value for matching SpatialSizeUnit
                    return self.Model.convertUnit( aSpecies[6], aValue )

                else:
                    aCompartmentUnit = self.Model.CompartmentUnit[aSpecies[2]]

                    if ( aCompartmentUnit != '' ):

                        # convert Value for matching CompartmentUnit
                        return self.Model.convertUnit( aCompartmentUnit,
                                                       aValue )
                    else:
                        return aValue 

            else:
                raise ValueError, "Value must be defined as InitialAmount or InitialConcentration"


    
    # =========================================================

    def getConstant( self, aSpecies ):

        if ( self.Model.Level == 1 ):

            if ( aSpecies[9] == 1 ):

                return aSpecies[9]

            else:
                return 0
            
        elif ( self.Model.Level == 2 ):

            if ( aSpecies[11] == 1 ):
                
                return aSpecies[11]

            else:
                return 0

    # =========================================================


# --------------------------------
# Rule Class
# --------------------------------

class SBML_Rule( SBML_Model ):

    def __init__( self, aModel ):

        self.Model = aModel
        self.RuleNumber = 0


    # =========================================================

    def initialize( self ):

        self.VariableReferenceList = []
        self.VariableNumber = 0
        self.ParameterNumber = 0
        self.RuleNumber = self.RuleNumber + 1


    # =========================================================
    
    def getRuleID( self ):

        return 'Process:/SBMLRule:Rule' + str( self.RuleNumber )


    # =========================================================

    def getVariableType( self, aName ):

        for aSpecies in self.Model.SpeciesList:

            if ( ( self.Model.Level == 1 and aSpecies[1] == aName ) or
                 ( self.Model.Level == 2 and aSpecies[0] == aName ) ):

                return libsbml.SBML_SPECIES

        for aParameter in self.Model.ParameterList:

            if ( ( self.Model.Level == 1 and aParameter[1] == aName ) or
                 ( self.Model.Level == 2 and aParameter[0] == aName ) ):

                return libsbml.SBML_PARAMETER

        for aCompartment in self.Model.CompartmentList:

            if ( ( self.Model.Level == 1 and aCompartment[1] == aName ) or
                 ( self.Model.Level == 2 and aCompartment[0] == aName ) ):

                return libsbml.SBML_COMPARTMENT

        raise TypeError, "Variable type must be Species, Parameter, or Compartment"
    

    # =========================================================

    def setMolarConcName( self, aName ):

        for aVariableReference in self.VariableReferenceList:
            if aVariableReference[1].split(':')[2] == aName:
                
                return aVariableReference[0] + '.MolarConc'
                

    # =========================================================

    def setValueName( self, aName ):

        for aVariableReference in self.VariableReferenceList:
            if aVariableReference[1].split(':')[2] == aName:

                return aVariableReference[0] + '.Value'


    # =========================================================

    def setSizeName( self, aName ):

        for aVariableReference in self.VariableReferenceList:
            if aVariableReference[1].split(':')[2] == 'SIZE':

                return aVariableReference[0] + '.Value'


    # =========================================================

    def setSpeciesToVariableReference( self, aName, aStoichiometry='0' ):

        for aSpecies in self.Model.SpeciesList:

            if ( ( self.Model.Level == 1 and aSpecies[1] == aName ) or
                 ( self.Model.Level == 2 and aSpecies[0] == aName ) ):
            
                for aVariableReference in self.VariableReferenceList:
                    if aVariableReference[1].split(':')[2] == aName:

                        return 

                aVariableList = []
                aVariableList.append( 'V' + str( self.VariableNumber ) )
                self.VariableNumber = self.VariableNumber + 1
                
                aVariableID = self.Model.getSpeciesReferenceID( aName )
                aVariableList.append( 'Variable:' + aVariableID )
                aVariableList.append( aStoichiometry )
                
                self.VariableReferenceList.append( aVariableList )
                

    # =========================================================

    def setParameterToVariableReference( self, aName, aStoichiometry='0' ):

        for aParameter in self.Model.ParameterList:

            if ( ( self.Model.Level == 1 and aParameter[1] == aName ) or
                 ( self.Model.Level == 2 and aParameter[0] == aName ) ):
                
                for aVariableReference in self.VariableReferenceList:
                    if aVariableReference[1].split(':')[2] == aName:

                        return

                aParameterList = []
                aParameterList.append( 'P' + str( self.ParameterNumber ) )
                self.ParameterNumber = self.ParameterNumber + 1
                aParameterList.append( 'Variable:/SBMLParameter:' + aName )
                aParameterList.append( aStoichiometry )
                self.VariableReferenceList.append( aParameterList )
            

    # =========================================================

    def setCompartmentToVariableReference( self, aName, aStoichiometry='0' ):

        for aCompartment in self.Model.CompartmentList:

            if ( ( self.Model.Level == 1 and aCompartment[1] == aName ) or
                 ( self.Model.Level == 2 and aCompartment[0] == aName ) ):
                
                print self.VariableReferenceList
                for aVariableReference in self.VariableReferenceList:
                    if ( aVariableReference[1].split(':')[1] ==\
                       self.Model.getPath( aName ) ) and\
                    ( aVariableReference[1].split(':')[2] == 'SIZE' ):

                        return

                aCompartmentList = []
                aCompartmentList.append( aName )
                
                aCompartmentList.append(
                    'Variable:' + self.Model.getPath( aName ) + ':SIZE' )
                
                aCompartmentList.append( aStoichiometry )
                self.VariableReferenceList.append( aCompartmentList )
                

                            
    # =========================================================

    def __convertVariableName( self, anASTNode ):

        aNumChildren = anASTNode.getNumChildren()

        if ( aNumChildren == 2 ):
            self.__convertVariableName( anASTNode.getLeftChild() )
            self.__convertVariableName( anASTNode.getRightChild() )

        elif ( aNumChildren == 1 ):
            self.__convertVariableName( anASTNode.getLeftChild() )

        elif ( aNumChildren == 0 ):
            if ( anASTNode.isNumber() == 1 ):
                pass

            else:
                aName = anASTNode.getName()
                newName = []
                aType = self.getVariableType( aName )

                # Species
                if ( aType == libsbml.SBML_SPECIES ):

                    self.setSpeciesToVariableReference( aName )
                    newName.append( self.setMolarConcName( aName ) )
                    if( newName[0] != '' ):
                        anASTNode.setName( newName[0] )      
                        return anASTNode

                # Parameter
                if ( aType == libsbml.SBML_PARAMETER ):
                    
                    self.setParameterToVariableReference( aName )
                    newName.append( self.setValueName( aName ) )
                    if( newName[0] != '' ):
                        anASTNode.setName( newName[0] )                    
                        return anASTNode

                # Compartment
                if ( aType == libsbml.SBML_COMPARTMENT ):
                    
                    self.setCompartmentToVariableReference( aName )
                    newName.append( self.setSizeName( aName ) )
#                    print newName
                    if( newName[0] != '' ):
                        anASTNode.setName( newName[0] )                    
                        return anASTNode

        return anASTNode


    # =========================================================

    def convertRuleFormula( self, aFormula ):

        aASTRootNode = libsbml.parseFormula( aFormula )

        convertedAST = self.__convertVariableName( aASTRootNode )
        convertedFormula = libsbml.formulaToString( convertedAST )
        
        return convertedFormula


    # =========================================================




# --------------------------------
# Reaction Class
# --------------------------------    

class SBML_Reaction( SBML_Model ):
    

    def __init__( self, aModel ):
        
        self.Model = aModel


    # =========================================================
    
    def initialize( self ):

        self.SubstrateNumber = 0
        self.ProductNumber = 0
        self.ModifierNumber = 0
        self.ParameterNumber = 0

        self.VariableReferenceList = []


    # =========================================================
    
    def getReactionID( self, aReaction ):

        if ( self.Model.Level == 1 ):
            if ( aReaction[1] != '' ):
                return 'Process:/:' + aReaction[1]
            else:
                raise NameError,"Reaction must set the Reaction name"
                
        elif ( self.Model.Level == 2 ):
            if ( aReaction[0] != '' ):
                return 'Process:/:' + aReaction[0]
            else:
                raise NameError,"Reaction must set the Reaction ID"


    # =========================================================

    def __convertVariableName( self, anASTNode ):
        
        aNumChildren = anASTNode.getNumChildren()

        if ( aNumChildren == 2 ):

            #if ( anASTNode.isFunction() ):

                # Macro expand
                #if( self.Model.FunctionDefinition[ anASTNode.getName() ] != None ):
                #    self.Model.macroExpand( anASTNode )


            self.__convertVariableName( anASTNode.getLeftChild() )
            self.__convertVariableName( anASTNode.getRightChild() )

            return anASTNode
        

        elif ( aNumChildren == 1 ):

            #if ( anASTNode.isFunction() ):

                # Macro expand
                #if( self.Model.FunctionDefinition[ anASTNode.getName() ] != None ):
                #    self.Model.macroExpand( anASTNode )

                
            self.__convertVariableName( anASTNode.getLeftChild() )

            return anASTNode
        

        elif ( aNumChildren == 0 ):
            if ( anASTNode.isNumber() == 1 ):
                pass
            else:
                aName = anASTNode.getName()
                newName = []

                for aSpecies in self.Model.SpeciesList:
                    if ( aSpecies[0] == aName or aSpecies[1] == aName):

                        for aVariableReference in self.VariableReferenceList:
                            if aVariableReference[1].split(':')[2] == aName:
                                newName.append( aVariableReference[0] + '.MolarConc' )
                            else:
                                pass

                        if( self.Model.Level == 2 and newName == [] ):
                            raise NameError,"in libSBML :",aName,"isn't defined in VariableReferenceList"

                        elif( self.Model.Level == 1 and newName == [] ):

                            aModifierList = []
                            aModifierList.append(
                                'C' + str( self.ModifierNumber ) )
                            self.ModifierNumber = self.ModifierNumber + 1
                            
                            aModifierID = self.Model.getSpeciesReferenceID( aName )
                            aModifierList.append( 'Variable:' + aModifierID )
                            aModifierList.append( '0' )
                            self.VariableReferenceList.append( aModifierList )
                            
                            newName.append( aModifierList[0] + '.MolarConc' )

                        anASTNode.setName( newName[0] )      

                        return anASTNode


                if ( newName == [] ):

                    for aParameter in self.Model.ParameterList:
                        if ( aParameter[0] == aName or
                             aParameter[1] == aName ):

                            for aVariableReference in self.VariableReferenceList:
                                if aVariableReference[1].split(':')[2] == aName:
                                    newName.append( aVariableReference[0] + '.Value' )
                            else:
                                pass

                            if( newName == [] ):

                                aParameterList = []
                                aParameterList.append(
                                    'Param' + str( self.ParameterNumber ) )
                            
                                self.ParameterNumber = self.ParameterNumber + 1

                                aParameterList.append(
                                    'Variable:/SBMLParameter:' + aName )
                            
                                aParameterList.append( '0' )
                                self.VariableReferenceList.append( aParameterList )

                                newName.append( aParameterList[0] + '.Value' )


                            anASTNode.setName( newName[0] )

                            return anASTNode

                    
                if ( newName == [] ):
                    for aCompartment in self.Model.CompartmentList:
                        if ( aCompartment[0] == aName or
                             aCompartment[1] == aName ):

                            for aVariableReference in self.VariableReferenceList:
                                if( aVariableReference[1].split(':')[2] == 'SIZE' ):
                                    aCurrentPath = ( aVariableReference[1].split(':')[1] )
                                    aLastSlash = string.rindex( aCurrentPath, '/' )
                                    newName.append( aCurrentPath[aLastSlash+1:] + '.Value' )
                                else:
                                    pass

                            if( newName == [] ):
                                
                                aCompartmentList = []
                                aCompartmentList.append( aName )
                            
                                aCompartmentList.append(
                                    'Variable:' + self.Model.getPath( aName ) + ':SIZE' )
                            
                                aCompartmentList.append( '0' )
                                self.VariableReferenceList.append( aCompartmentList )

                                newName.append( aCompartmentList[0] + '.Value' )


                            anASTNode.setName( newName[0] )                    
                            
                            return anASTNode

            
                return anASTNode
                

    # =========================================================
    
    def convertKineticLawFormula( self, aFormula ):

        aASTRootNode = libsbml.parseFormula( aFormula )
        convertedAST = self.__convertVariableName( aASTRootNode )

        if( self.VariableReferenceList[0][0] == "S0" ):

            aConvertedFormula = "( " + libsbml.formulaToString( convertedAST ) + " )" + " * S0.getSuperSystem().SizeN_A"
                
        elif( self.VariableReferenceList[0][0] == "P0" ):
            
            aConvertedFormula = "( " + libsbml.formulaToString( convertedAST ) + " )" + " * P0.getSuperSystem().SizeN_A"
            
        elif( self.VariableReferenceList[0][0] == "C0" ):

            aConvertedFormula = "( " + libsbml.formulaToString( convertedAST ) + " )" +" * C0.getSuperSystem().SizeN_A" 

        else:
            aConvertedFormula = libsbml.formulaToString( convertedAST )


        return aConvertedFormula


    # =========================================================

    def getStoichiometry( self, aSpeciesID, aStoichiometry ):

        if ( self.Model.Level == 1 ):
            for aSpecies in self.Model.SpeciesList:
                if ( aSpecies[1] == aSpeciesID ):
                    if( aSpecies[9] == 1 ):
                        return int( 0 )
                    else:
                        return int( aStoichiometry )

        elif ( self.Model.Level == 2 ):
            for aSpecies in self.Model.SpeciesList:
                if ( aSpecies[0] == aSpeciesID ):
                    if( aSpecies[11] == 1 ):
                        return int( 0 )
                    else:
                        return int( aStoichiometry )

        else:
           raise Exception,"Version",self.Level," ????"


    # =========================================================





# --------------------------------
# Parameter Class
# --------------------------------    

class SBML_Parameter( SBML_Model ):


   def __init__( self, aModel ):
        self.Model = aModel


    # =========================================================

   def getParameterID( self, aParameter ):

       if ( self.Model.Level == 1 ):
           if ( aParameter[1] != '' ):
               return 'Variable:/SBMLParameter:' + aParameter[1]
           else:
               raise NameError, "Parameter must set the Parameter Name"

       elif ( self.Model.Level == 2 ):
           if ( aParameter[0] != '' ):
               return 'Variable:/SBMLParameter:' + aParameter[0]
           else:
               raise NameError, "Parameter must set the Parameter ID"

       else:
           raise Exception,"Version",self.Level," ????"
                

   # =========================================================

   def getParameterValue( self, aParameter ):
       
       if ( aParameter[3] != '' and aParameter[2] != 0 ):

           return self.Model.convertUnit( aParameter[3], aParameter[2] )
       
       else:

           return aParameter[2]
        

    # =========================================================



# --------------------------------
# Event Class
# --------------------------------    

class SBML_Event( SBML_Model ):

    def __init__( self, aModel ):
        self.Model = aModel
        self.EventNumber = 0


    # =========================================================

    def getEventID( self, aEvent ):

        if( aEvent[0] != '' ):
            return 'Process:/:' + aEvent[0]
        elif( aEvent[1] != '' ):
            return 'Process:/:' + aEvent[1]
        else:
            anID = 'Process:/:Event' + self.EventNumber
            self.EventNumber = self.EventNumber + 1
            return anID

    # =========================================================
