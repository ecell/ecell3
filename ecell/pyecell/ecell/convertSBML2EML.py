from libsbml import *
from SbmlFunctions import *

import sys

#if __name__ == '__main__':

#    import sys
    
#    filename = sys.argv[1]
#    aSBMLdocument = readSBML( filename )
#    aSBMLmodel = SBMLDocument_getModel( aSBMLdocument )

    
''' Compartment List '''

#   " [[ Id , Name , SpatialDimension , Size , Volume , Unit , Ouside , Constant ]] "

    
''' Event List '''

#   " [[ Id , Name , StringTrigger , StringDelay , TimeUnit , [[ VariableAssignment , StringAssignment ]] ]] "

    
''' FunctionDefinition List '''

#   " [[ Id , Name , String ]] "
    

''' Parameter List '''

#   " [[ Id , Name , Value , Unit , Constant ]] "
    

''' Reaction List '''

#   " [[ Id , Name , [ Formula , String , TimeUnit , SubstanceUnit , [[ ParameterId , ParameterName , ParameterValue , ParameterUnit , ParameterConstant ]] ] , Reversible , Fast , [[ ReactantSpecies , ( ReactantStoichiometry , ReactantStoichiometryMath ) , ReactantDenominator  ]] , [[  ProductSpecies , ( ProductStoichiometry , ProductStoichiometryMath ) , ProductDenominator ]] , [[ ModifierSpecies ]] ]] "

    
''' Rule List '''

#   " [[ Formula ]] "


''' Species List '''

#   " [[ Id , Name , Compartment , InitialAmount , InitialConcentration , SubstanceUnit , SpatialSizeUnit , Unit , HasOnlySubstanceUnit , BoundaryCondition , Charge , Constant ]] "
    

''' UnitDefinition List '''

#   " [[ Id , Name , [[ Kind , Exponent , Scale , Multiplier , Offset ]] ]] "
    

# Avogadro Number set
N_A = 6.0221367e+23



class SBML_Model:

    def __init__( self, aSBMLDocument, aSBMLmodel ):

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

        self.CompartmentSize = {}
        self.CompartmentUnit = {}

    def getPath( self, aCompartmentID ):
        for aCompartment in self.CompartmentList:
            if ( self.Level == 1 ):
                if ( aCompartment[1] == aCompartmentID ):
                    if ( aCompartment[6] == '' ):
                        aPath = '/' + aCompartmentID
                        return aPath
                    else:
                        aPath = self.getPath( aCompartment[6] ) + '/' + aCompartmentID
                        return aPath

            elif( self.Level == 2 ):
                if( aCompartment[0] == aCompartmentID ):
                    if ( aCompartment[6] == '' ):
                        aPath = '/' + aCompartmentID
                        return aPath
                    else:
                        aPath = self.getPath( aCompartment[6] ) + '/' + aCompartmentID
                        return aPath

            else:
                print "Version",self.Level," ????"
                sys.exit(1)



    def getSpeciesReferenceID( self, aSpeciesID ):

        for aSpecies in self.SpeciesList: 
            if ( self.Level == 1 ):
                if ( aSpecies[1] == aSpeciesID ):
                    return self.getPath( aSpecies[2] ) + ":" + aSpeciesID
                    
            elif ( self.Level == 2 ):
                if ( aSpecies[0] == aSpeciesID ):
                    return self.getPath( aSpecies[2] ) + ":" + aSpeciesID

            else:
                print "Version",self.Level," ????"
                sys.exit(1)



class SBML_Compartment( SBML_Model ):


    def __init__( self, aModel ):
        self.Model = aModel

    def getCompartmentID( self, aCompartment ):
        
        if ( aCompartment[6] == '' ):
            if ( self.Model.Level == 1 ):
                aSystemID = '/:' + aCompartment[1]
            elif ( self.Model.Level == 2 ):
                aSystemID = '/:' + aCompartment[0]
            else:
                print "Compartment Class needs a ['ID']"
                sys.exit(1)
        else:
            if( self.Model.Level == 1 ):
                aSystemID = self.Model.getPath( aCompartment[6] ) + ':'+ aCompartment[1]
            elif( self.Model.Level == 2 ):
                aSystemID = self.Model.getPath( aCompartment[6] ) + ':'+ aCompartment[0]

        return 'System:' + aSystemID


    def setCompartmentSize_Unit( self, aCompartment ):

        if( self.Model.Level == 1 ):
            self.Model.CompartmentSize[ aCompartment[1] ] = aCompartment[4]
            self.Model.CompartmentUnit[ aCompartment[1] ] = aCompartment[5]

        elif( self.Model.Level == 2 ):
            if( aCompartment[3] != "Unknown" ):
                self.Model.CompartmentSize[ aCompartment[0] ] = aCompartment[3]
                self.Model.CompartmentUnit[ aCompartment[0] ] = aCompartment[5]

            elif( aCompartment[3] == "Unknown" ):
                [ self.Model.CompartmentSize[ aCompartment[0] ], self.Model.CompartmentUnit[ aCompartment[0] ] ] = self.__getCompartmentOutsideSize_Unit( aCompartment )


    def __getCompartmentOutsideSize_Unit( self, aCompartment ):

        if( aCompartment[3] == "Unknown" and aCompartment[6] == '' ):
            return [ 1, aCompartment[5] ]
        elif( aCompartment[3] != "Unknown" ):
            return [ aCompartment[3], aCompartment[5] ]
        elif( aCompartment[3] == "Unknown" and aCompartment[6] != '' ):
            for tmpCompartment in self.Model.CompartmentList:
                if ( tmpCompartment[0] == aCompartment[6] ):
                    return self.__getCompartmentOutsideSize_Unit( tmpCompartment )
        else:
            print "Unexpected error"
            sys.exit(1)
        

    def convertCompartmentUnit( self, aSBMLModel, aCompartmentSize, aCompartmentUnit ):

        convertedValueList = []
        for anUnitDefinition in self.Model.UnitDefinitionList:
            if ( anUnitDefinition[0] == aCompartmentUnit ):
                for anUnit in anUnitDefinition[2]:

                    # litre, metre,
                    if( anUnit[0] == "litre" or anUnit[0] == "metre" ):
                        tmpValue = pow( aCompartmentSize, anUnit[1] ) * pow( 10, anUnit[2] ) * anUnit[3] + anUnit[4]
                        convertedValueList.append( tmpValue )

        if ( convertedValueList == [] ):
            convertedValueList.append( aCompartmentSize )
            
        return convertedValueList[0]



class SBML_Species( SBML_Model ):

    def __init__( self, aModel ):
        self.Model = aModel
    
    def getSpeciesID( self, aSpecies ):

        aCompartmentID = aSpecies[2]

        if ( self.Model.Level == 1 ):
            aSystemID = self.Model.getPath( aCompartmentID ) + ':' + aSpecies[1]
    
        elif ( self.Model.Level == 2 ):
            aSystemID = self.Model.getPath( aCompartmentID ) + ':' + aSpecies[0]
        else:
            print "Version",self.Level," ????"
            sys.exit(1)
                
        return 'Variable:' + aSystemID


    def getSpeciesValue( self, aSpecies ):

        if ( self.Model.Level == 1 ):
            if ( aSpecies[3] != "Unknown" ):
                aValue = self.__convertSpeciesValue( aSpecies[3], aSpecies )
            else:
                print aSpecies[1] + " : InitialAmount must be defined, but this model is undefined."
                sys.exit(1)


        elif ( self.Model.Level == 2 ):
            if ( aSpecies[3] != "Unknown" ):
                aValue = self.__convertSpeciesValue( aSpecies[3], aSpecies )

            elif ( aSpecies[4] != "Unknown" ):
                aSize = self.Model.CompartmentSize[ aSpecies[2] ]
                aSize = self.__convertSpeciesSize( aSize, aSpecies )

                aValue = aSpecies[4] * aSize
                aValue = self.__convertSpeciesValue( aValue, aSpecies )

            else:
                aValue = "Unknown"

        return aValue

    

    def __convertSpeciesValue( self, aValue, aSpecies ):

        # Need error check in here!!
        
        if ( self.Model.Level == 1 ):
            if( aSpecies[7] != '' ):
                if( aSpecies[7] == "item" ):
                    return aValue
                else:
                    return self.__convertSpeciesUnit( aValue, aSpecies[7] )
            else:
                return aValue * N_A
            
        elif ( self.Model.Level == 2 ):

            if( aSpecies[5] != '' ):
                if( aSpecies[5] == "item" ):
                    return aValue
                else:
                    return self.__convertSpeciesUnit( aValue, aSpecies[5] )
            else:
                return aValue * N_A


    def __convertSpeciesSize( self, aSize, aSpecies ):

        if ( aSpecies[6] != '' ):
            for aCompartment in self.Model.CompartmentList:
                if( aCompartment[0] == aSpecies[2] ):
                    if( aCompartment[2] == 0 or aSpecies[8] == 1 ):
                        print "Error : SpatialSizeUnits must not have a value if spatialDimensions on the compartment has a value of 0, or if the species's value of the units field of the species' compartment"
                        sys.exit(1)

            return self.__convertSpeciesUnit( aSize, aSpecies[6] )

        elif ( self.Model.CompartmentUnit[ aSpecies[2] ] != '' ):
            
            return self.__convertSpeciesUnit( aSize, self.Model.CompartmentUnit[ aSpecies[2] ] )

        else:
            return aSize


    def __convertSpeciesUnit( self, aValue, anUnitID ):

        convertedValueList = []
        for anUnitDefinition in self.Model.UnitDefinitionList:
            if ( anUnitDefinition[0] == anUnitID ):
                for anUnit in anUnitDefinition[2]:
                    # mole ( convert to item )
                    if( anUnit[0] == "mole" ):
                        tmpValue = ( pow( aValue, anUnit[1] ) * pow( 10, anUnit[2] ) * anUnit[3] + anUnit[4] ) * N_A
                        convertedValueList.append( tmpValue )

                    # item, litre, metre
                    elif ( anUnit[0] == "item" or anUnit[0] == "litre" or anUnit[0] == "metre" ):
                        tmpValue = ( pow( aValue, anUnit[1] ) * pow( 10, anUnit[2] ) * anUnit[3] + anUnit[4] )
                        convertedValueList.append( tmpValue )

                    else:
                        print "Unit Error : SBML Importer isn't mounted ",anUnit[0]," Unit"
                        sys.exit(1)

        if ( convertedValueList == [] ):
            if ( anUnitID == "mole" ):
                
                tmpValue = aValue * N_A
                convertedValueList.append( tmpValue )
                
            else:
                print "Unit Error : SBML Importer isn't mounted ",anUnit[0]," Unit"
                sys.exit(1)

        return convertedValueList[0]



class SBML_Rule( SBML_Model ):

    def __init__( self, aModel ):

        self.Model = aModel
        self.RuleNumber = 0

    def initialize( self ):

        self.VariableReferenceList = []
        self.VariableNumber = 0
        self.ParameterNumber = 0
        self.RuleNumber = self.RuleNumber + 1

    def getRuleID( self ):

        return 'Process:/:Rule' + str( self.RuleNumber )


    def __convertVariableName( self, aASTNode ):

        aNumChildren = aASTNode.getNumChildren()

        if ( aNumChildren == 2 ):
            self.__convertVariableName( aASTNode.getLeftChild() )
            self.__convertVariableName( aASTNode.getRightChild() )

        elif ( aNumChildren == 1 ):
            self.__convertVariableName( aASTNode.getLeftChild() )

        elif ( aNumChildren == 0 ):
            if ( aASTNode.isNumber() == 1 ):
                pass
            else:
                aName = aASTNode.getName()
                newName = []
                for aSpecies in self.Model.SpeciesList:
                    if ( aSpecies[0] == aName or aSpecies[1] == aName):
                        aVariableList = []
                        aVariableList.append( 'Variable' + str( self.VariableNumber ) )
                        self.VariableNumber = self.VariableNumber + 1
                        
                        aVariableID = self.Model.getSpeciesReferenceID( aName )
                        aVariableList.append( 'Variable:' + aVariableID )
                        aVariableList.append( '0' )

                        self.VariableReferenceList.append( aVariableList )
                        
                        newName.append( aVariableList[0] + '.Value' )
                        
                        aASTNode.setName( newName[0] )      

                if ( newName == [] ):
                    for aParameter in self.Model.ParameterList:
                        if ( aParameter[0] == aName or aParameter[1] == aName ):
                            aParameterList = []
                            aParameterList.append( 'Parameter' + str( self.ParameterNumber ) )
                            self.ParameterNumber = self.ParameterNumber + 1
                            aParameterList.append( 'Variable:/SBMLParameter:' + aName )
                            aParameterList.append( '0' )
                            self.VariableReferenceList.append( aParameterList )

                            newName.append( aParameterList[0] + '.Value' )

                            aASTNode.setName( newName[0] )
                    
        return aASTNode


    
    def convertRuleFormula( self, aFormula ):

        aASTRootNode = parseFormula( aFormula )
        convertedAST = self.__convertVariableName( aASTRootNode )
        convertedFormula = formulaToString( convertedAST )
        
        return convertedFormula




class SBML_Reaction( SBML_Model ):
    

    def __init__( self, aModel ):
        
        self.Model = aModel

    def initialize( self ):

        self.SubstrateNumber = 0
        self.ProductNumber = 0
        self.ModifierNumber = 0
        self.ParameterNumber = 0

        self.VariableReferenceList = []

    def getReactionID( self, aReaction ):

        if ( self.Model.Level == 1 ):
            if ( aReaction[1] != '' ):
                return 'Process:/:' + aReaction[1]
            else:
                print "Name Error: Reaction must set the Reaction name"
                sys.exit(1)
                
        elif ( self.Model.Level == 2 ):
            if ( aReaction[0] != '' ):
                return 'Process:/:' + aReaction[0]
            else:
                print "Name Error: Reaction must set the Reaction ID"
                sys.exit(1)

    def __convertVariableName( self, aASTNode ):

        aNumChildren = aASTNode.getNumChildren()

        if ( aNumChildren == 2 ):
            self.__convertVariableName( aASTNode.getLeftChild() )
            self.__convertVariableName( aASTNode.getRightChild() )

        elif ( aNumChildren == 1 ):
            self.__convertVariableName( aASTNode.getLeftChild() )

        elif ( aNumChildren == 0 ):
            if ( aASTNode.isNumber() == 1 ):
                pass
            else:
                aName = aASTNode.getName()
                newName = []
                for aSpecies in self.Model.SpeciesList:
                    if ( aSpecies[0] == aName or aSpecies[1] == aName):
                        for aVariableReference in self.VariableReferenceList:
                            if aVariableReference[1].split(':')[2] == aName:
                                newName.append( aVariableReference[0] + '.MolarConc' )
                            else:
                                pass

                        if( self.Model.Level == 2 and newName == [] ):
                            print "NameError in libSBML :",aName,"isn't defined in VariableReferenceList"
                            sys.exit(1)

                        elif( self.Model.Level == 1 and newName == []):
                            aModifierList = []
                            aModifierList.append(
                                'C' + str( self.ModifierNumber ) )
                            self.ModifierNumber = self.ModifierNumber + 1
                            
                            aModifierID = self.Model.getSpeciesReferenceID( aName )
                            aModifierList.append( 'Variable:' + aModifierID )
                            aModifierList.append( '0' )
                            self.VariableReferenceList.append( aModifierList )
                            
                            newName.append( aModifierList[0] + '.MolarConc' )

                        aASTNode.setName( newName[0] )      

                if ( newName == [] ):
                    for aParameter in self.Model.ParameterList:
                        if ( aParameter[0] == aName or
                             aParameter[1] == aName ):
                            
                            aParameterList = []
                            aParameterList.append(
                                'Parameter' + str( self.ParameterNumber ) )
                            
                            self.ParameterNumber = self.ParameterNumber + 1

                            aParameterList.append(
                                'Variable:/SBMLParameter:' + aName )
                            
                            aParameterList.append( '0' )
                            self.VariableReferenceList.append( aParameterList )

                            newName.append( aParameterList[0] + '.Value' )

                            aASTNode.setName( newName[0] )
                    
        return aASTNode


    
    def convertKineticLaw( self, aFormula ):

        aASTRootNode = parseFormula( aFormula )
        convertedAST = self.__convertVariableName( aASTRootNode )

        convertedFormula = []

        if( self.VariableReferenceList[0][0] == "S0" ):
            convertedFormula.append(
                "( " + formulaToString( convertedAST ) + " )" +
                " * S0.getSuperSystem().SizeN_A" )
            
        elif( self.VariableReferenceList[0][0] == "P0" ):
            convertedFormula.append(
                "( " + formulaToString( convertedAST ) + " )" +
                " * P0.getSuperSystem().SizeN_A" )
            
        elif( self.VariableReferenceList[0][0] == "C0" ):
            convertedFormula.append(
                "( " + formulaToString( convertedAST ) + " )" +
                " * C0.getSuperSystem().SizeN_A" )
        else:
            pass

        if( convertedFormula == [] ):
            convertedFormula.append( formulaToString( convertedAST ) )
            
        return convertedFormula[0]


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
           print "Version",self.Level," ????"
           sys.exit(1)


class SBML_Parameter( SBML_Model ):


   def __init__( self, aModel ):
        self.Model = aModel

   def getParameterID( self, aParameter ):
       if ( self.Model.Level == 1 ):
           if ( aParameter[1] != '' ):
               return 'Variable:/SBMLParameter:' + aParameter[1]
           else:
               print "Name Error: Parameter must set the Parameter Name"
               sys.exit(1)
       elif ( self.Model.Level == 2 ):
           if ( aParameter[0] != '' ):
               return 'Variable:/SBMLParameter:' + aParameter[0]
           else:
               print "Name Error: Parameter must set the Parameter ID"
               sys.exit(1)
       else:
           print "Version",self.Level," ????"
           sys.exit(1)
                

class SBML_Event( SBML_Model ):

    def __init__( self, aModel ):
        self.Model = aModel
        self.EventNumber = 0

    def getEventID( self, aEvent ):
        if( aEvent[0] != '' ):
            return 'Process:/:' + aEvent[0]
        elif( aEvent[1] != '' ):
            return 'Process:/:' + aEvent[1]
        else:
            anID = 'Process:/:Event' + self.EventNumber
            self.EventNumber = self.EventNumber + 1
            return anID
