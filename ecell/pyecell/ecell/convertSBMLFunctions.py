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

        self.setFunctionDefinitionToDictionary()
        

    # =========================================================

    def setFunctionDefinitionToDictionary( self ):

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
            raise Exception,"Version"+str(self.Level)+" ????"
        
        
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
            raise Exception,"Version"+str(self.Level)+" ????"


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
            raise Exception,"Version"+str(self.Level)+" ????"
                
        return 'Variable:' + aSystemID


    # =========================================================
    
    def getSpeciesValue( self, aSpecies ):

        if aSpecies[ 3 ] != 'Unknown': # initialAmount
            return float( aSpecies[ 3 ] )

        elif self.Model.Level == 2 \
                 and aSpecies[ 4 ] != 'Unknown': # initialConcentration

            # spatialSizeUnits and hasOnlySubstanceUnits should be checked
            
            aSize = self.Model.CompartmentSize[ aSpecies[ 2 ] ]
            return aSpecies[ 4 ] * aSize

        else:
            raise ValueError, 'InitialAmount or InitialConcentration of Species [%s] must be defined.' % ( aSpecies[ 0 ] )

##         if ( self.Model.Level == 1 ):

##             if ( aSpecies[3] != "Unknown" ):

##                 if ( aSpecies[7] == '' or aSpecies[7] == 'mole' ):

##                     return float( aSpecies[3] ) * N_A

##                 elif ( aSpecies[7] == 'item' ):

##                     return float( aSpecies[3] )

##                 else:
                    
##                     return ( self.Model.convertUnit( aSpecies[7], aSpecies[3] ) ) * N_A

##             else:
##                 raise ValueError, "InitialAmount must be defined, but this model is undefined."


##         elif ( self.Model.Level == 2 ):

##             if ( aSpecies[3] != "Unknown" ):

##                 if ( aSpecies[5] != '' and aSpecies[5] != 'mole' ):

##                     return ( self.Model.convertUnit( aSpecies[5], aSpecies[3] ) ) * N_A

##                 elif( aSpecies[5] == 'item' ):

##                     return float( aSpecies[3] )

##                 else:
##                     return float( aSpecies[3] ) * N_A
                

##             # aSpecies[4] : InitialConcentration
##             elif ( aSpecies[4] != "Unknown" ):

##                 aValue = aSpecies[4]
##                 aSize = self.Model.CompartmentSize[aSpecies[2]]

##                 # convert InitialConcentration into molecules number for E-Cell unit
##                 aValue = aValue * aSize * N_A

                
##                 # aSpecies[5] : SubstanceUnit
##                 if ( aSpecies[5] != '' ):

##                     # convert Value for matching SubstanceUnit
##                     aValue = self.Model.convertUnit( aSpecies[5], aValue )


##                 # aSpecies[8] : hasOnlySubstanceUnit
##                 if ( aSpecies[6] != '' ):

##                     # convert Value for matching SpatialSizeUnit
##                     return self.Model.convertUnit( aSpecies[6], aValue )

##                 else:
##                     aCompartmentUnit = self.Model.CompartmentUnit[aSpecies[2]]

##                     if ( aCompartmentUnit != '' ):

##                         # convert Value for matching CompartmentUnit
##                         return self.Model.convertUnit( aCompartmentUnit,
##                                                        aValue )
##                     else:
##                         return aValue 

##             else:
##                 raise ValueError, "Value must be defined as InitialAmount or InitialConcentration"


    
    # =========================================================

    def getConstant( self, aSpecies ):

        if ( self.Model.Level == 1 ):

            if ( aSpecies[9] == 1 ):

                return 1

            else:
                return 0
            
        elif ( self.Model.Level == 2 ):

            if ( aSpecies[11] == 1 ):
                
                return 1

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
    

##     # =========================================================

##     def setMolarConcName( self, aName ):

##         for aVariableReference in self.VariableReferenceList:
##             if aVariableReference[1].split(':')[2] == aName:
                
##                 return aVariableReference[0] + '.MolarConc'
                

##     # =========================================================

##     def setValueName( self, aName ):

##         for aVariableReference in self.VariableReferenceList:
##             if aVariableReference[1].split(':')[2] == aName:

##                 return aVariableReference[0] + '.Value'


##     # =========================================================

##     def setSizeName( self, aName ):

##         for aVariableReference in self.VariableReferenceList:
##             if aVariableReference[1].split(':')[2] == 'SIZE':

##                 return aVariableReference[0] + '.Value'


##     # =========================================================

    def setSpeciesToVariableReference( self, aName, aStoichiometry='0' ):

        for aSpecies in self.Model.SpeciesList:

            if ( ( self.Model.Level == 1 and aSpecies[1] == aName ) or
                 ( self.Model.Level == 2 and aSpecies[0] == aName ) ):
            
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

                aVariableID = self.Model.getSpeciesReferenceID( aName )
                aVariableList.append( 'Variable:' + aVariableID )
                aVariableList.append( aStoichiometry )
                
                self.VariableReferenceList.append( aVariableList )
                
                compartmentID = aSpecies[ 2 ]
                compartmentName = self.setCompartmentToVariableReference( compartmentID )

                return ( variableName, compartmentName )

    # =========================================================

    def setParameterToVariableReference( self, aName, aStoichiometry='0' ):

        for aParameter in self.Model.ParameterList:

            if ( ( self.Model.Level == 1 and aParameter[1] == aName ) or
                 ( self.Model.Level == 2 and aParameter[0] == aName ) ):
                
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

    # =========================================================

    def setCompartmentToVariableReference( self, aName, aStoichiometry='0' ):

        for aCompartment in self.Model.CompartmentList:

            if ( ( self.Model.Level == 1 and aCompartment[1] == aName ) or
                 ( self.Model.Level == 2 and aCompartment[0] == aName ) ):
                
                for c in range( len( self.VariableReferenceList ) ):
                    aVariableReference = self.VariableReferenceList[ c ]
                    
                    if ( aVariableReference[1].split(':')[1] ==\
                       self.Model.getPath( aName ) ) and\
                    ( aVariableReference[1].split(':')[2] == 'SIZE' ):

                        if aStoichiometry != 0:
                            aVariableReference[ 2 ] = aStoichiometry

                        return aVariableReference[ 0 ]

                aCompartmentList = []
                aCompartmentList.append( aName )
                
                aCompartmentList.append(
                    'Variable:' + self.Model.getPath( aName ) + ':SIZE' )
                
                aCompartmentList.append( aStoichiometry )
                self.VariableReferenceList.append( aCompartmentList )
                
                return aName
                            
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
                elif ( aType == libsbml.SBML_PARAMETER ):
                    
                    variableName = self.setParameterToVariableReference( aName )
                    if( variableName != '' ):
                        anASTNode.setName( '%s.Value' % ( variableName ) )
                        return anASTNode

                # Compartment
                elif ( aType == libsbml.SBML_COMPARTMENT ):
                    
                    variableName = self.setCompartmentToVariableReference( aName )
                    if( variableName != '' ):
                        anASTNode.setName( '%s.Value' % ( variableName ) )
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

    def setCompartmentToVariableReference( self, aName ):

        for aCompartment in self.Model.CompartmentList:
            if ( aCompartment[0] == aName or
                 aCompartment[1] == aName ):

                for aVariableReference in self.VariableReferenceList:
                    if( aVariableReference[1].split(':')[2] == 'SIZE' ):
                        aCurrentPath = ( aVariableReference[1].split(':')[1] )
                        aLastSlash = string.rindex( aCurrentPath, '/' )
                        variableName = aCurrentPath[aLastSlash+1:]
                        return aVariableReference[ 0 ]
                        ## return variableName
                                
                aCompartmentList = []
                aCompartmentList.append( aName )
                            
                aCompartmentList.append(
                    'Variable:' + self.Model.getPath( aName ) + ':SIZE' )
                            
                aCompartmentList.append( '0' )
                self.VariableReferenceList.append( aCompartmentList )

                return aCompartmentList[0]

        return ''
    
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
                variableName = ''

                for aSpecies in self.Model.SpeciesList:
                    if ( aSpecies[0] == aName or aSpecies[1] == aName):

                        for aVariableReference in self.VariableReferenceList:
                            if aVariableReference[1].split(':')[2] == aName:
                                variableName =  aVariableReference[0]

                        if( self.Model.Level == 2 and variableName == '' ):
                            raise NameError,"in libSBML :"+aName+" isn't defined in VariableReferenceList"

                        elif( self.Model.Level == 1 and variableName == '' ):

                            aModifierList = []
                            aModifierList.append(
                                'C' + str( self.ModifierNumber ) )
                            self.ModifierNumber = self.ModifierNumber + 1
                            
                            aModifierID = self.Model.getSpeciesReferenceID( aName )
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

##                if variableName == '':
                for aParameter in self.Model.ParameterList:
                    if ( aParameter[0] == aName or
                         aParameter[1] == aName ):

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

##                if variableName == '':
                variableName = self.setCompartmentToVariableReference( aName )
                if variableName != '':
                    anASTNode.setName( '%s.Value' % ( variableName ) )
                    return anASTNode
                
                return anASTNode

    # =========================================================
    
    def convertKineticLawFormula( self, aFormula ):

        aASTRootNode = libsbml.parseFormula( aFormula )
        convertedAST = self.__convertVariableName( aASTRootNode )

        return libsbml.formulaToString( convertedAST )
    
##         if( self.VariableReferenceList[0][0] == "S0" ):

##             aConvertedFormula = "( " + libsbml.formulaToString( convertedAST ) + " )" + " * S0.getSuperSystem().SizeN_A"
                
##         elif( self.VariableReferenceList[0][0] == "P0" ):
            
##             aConvertedFormula = "( " + libsbml.formulaToString( convertedAST ) + " )" + " * P0.getSuperSystem().SizeN_A"
            
##         elif( self.VariableReferenceList[0][0] == "C0" ):

##             aConvertedFormula = "( " + libsbml.formulaToString( convertedAST ) + " )" +" * C0.getSuperSystem().SizeN_A" 

##         else:
##             aConvertedFormula = libsbml.formulaToString( convertedAST )


##         return aConvertedFormula


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
           raise Exception,"Version"+str(self.Level)+" ????"


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
           raise Exception,"Version"+str(self.Level)+" ????"
                

   # =========================================================

   def getParameterValue( self, aParameter ):

       return aParameter[ 2 ]
   
##        if ( aParameter[3] != '' and aParameter[2] != 0 ):

##            return self.Model.convertUnit( aParameter[3], aParameter[2] )
       
##        else:

##            return aParameter[2]
        

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
