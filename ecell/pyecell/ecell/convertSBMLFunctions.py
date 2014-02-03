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
from SbmlFunctions import *

import sys

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
# Base Class
# --------------------------------

class SBML_Base:


    def __init__( self ):

        self.EntityPath = {
            'Parameter' : '/SBMLParameter',
            'Rule'      : '/SBMLRule',
            'Reaction'  : '/',
            'Event'     : '/SBMLEvent' }

    # =========================================================

    #def macroExpand( self, anASTNode ):

    #    aNumChildren = anASTNode.getNumChildren()

    #if ( aNumChildren == 2 ):

    #        aString = self.FunctionDefinition[ anASTNode.getName() ]

    #        anASTNode.getLeftChild().getName()
    #        anASTNode.getRightChild().getName

            
    #    if ( aNumChildren == 1 ):



    # =========================================================

    def getPath( self, aCompartmentID, aModel ):

        if( aCompartmentID == 'default' ):
            return '/'
        
        if ( aModel.Level == 1 ):
            for aCompartment in aModel.CompartmentList:
                if ( aCompartment[ 'Name' ] == aCompartmentID ):
                    if ( aCompartment[ 'Outside' ] == '' or\
                         aCompartment[ 'Outside' ] == 'default' ):
                        aPath = '/' + aCompartmentID
                        return aPath
                    else:
                        aPath = self.getPath( aCompartment[ 'Outside' ] ) + '/' +\
                                aCompartmentID
                        return aPath

        elif( aModel.Level >= 2 ):
            for aCompartment in aModel.CompartmentList:
                if( aCompartment[ 'Id' ] == aCompartmentID ):
                    if ( aCompartment[ 'Outside' ] == '' or\
                         aCompartment[ 'Outside' ] == 'default' ):
                        aPath = '/' + aCompartmentID
                        return aPath
                    else:
                        aPath = self.getPath( aCompartment[ 'Outside' ] ) + '/' +\
                                aCompartmentID
                        return aPath

        else:
            raise Exception,"Version"+str(self.Level)+" ????"
        
        
    # =========================================================

    def getSpeciesReferenceID( self, aSpeciesID, aModel ):

        if ( aModel.Level == 1 ):
            for aSpecies in aModel.SpeciesList:
                if ( aSpecies[ 'Name' ] == aSpeciesID ):
                    return self.getPath( aSpecies[ 'Compartment' ] ) + ":" + aSpeciesID
                    
        elif ( aModel.Level >= 2 ):
            for aSpecies in aModel.SpeciesList:
                if ( aSpecies[ 'Id' ] == aSpeciesID ):
                    return self.getPath( aSpecies[ 'Compartment' ] ) + ":" + aSpeciesID

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

        elif ( self.Level >= 2 ):
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

    def dic2FullID( self, anEntityDic ):

        return "%(Type)s:%(Path)s:%(EntityName)s" % anEntityDic

    # =========================================================

    def getVariableType( self, aName, aModel ):

        for aSpecies in aModel.SpeciesList:

            if ( ( aModel.Level == 1 and aSpecies[ 'Name' ] == aName ) or
                 ( aModel.Level >= 2 and aSpecies[ 'Id' ] == aName ) ):

                return libsbml.SBML_SPECIES

        for aParameter in aModel.ParameterList:

            if ( ( aModel.Level == 1 and aParameter[ 'Name' ] == aName ) or
                 ( aModel.Level >= 2 and aParameter[ 'Id' ] == aName ) ):

                return libsbml.SBML_PARAMETER

        for aCompartment in aModel.CompartmentList:

            if ( ( aModel.Level == 1 and aCompartment[ 'Name' ] == aName ) or
                 ( aModel.Level >= 2 and aCompartment[ 'Id' ] == aName ) ):

                return libsbml.SBML_COMPARTMENT

        raise TypeError, "Variable type must be Species, Parameter, or Compartment (got %s)" % aName

    # =========================================================
    
    def getID( self, anEntity, aModel ):

        if ( aModel.Level == 1 ):
            return anEntity[ 'Name' ]

        elif ( aModel.Level >= 2 ):
            return anEntity[ 'Id' ]

        else:
            raise Exception,"Version"+str(self.Level)+" ????"

    # =========================================================



# --------------------------------
# Model Class
# --------------------------------

class SBML_Model( SBML_Base ):


    def __init__( self, aSBMLDocument, aSBMLmodel ):

        SBML_Base.__init__( self )

        self.CompartmentSize = {}
        self.CompartmentUnit = {}
        self.FunctionDefinition = {}

        self.Level = aSBMLDocument.getLevel()
        self.Version = aSBMLDocument.getVersion() 

        self.DerivedValueDic = {}
        self.TimeSymbol = getTimeSymbol( aSBMLmodel )

        self.CompartmentList = getCompartment( aSBMLmodel )
        self.EventList = getEvent( aSBMLmodel, self.TimeSymbol )
        self.FunctionDefinitionList = getFunctionDefinition( aSBMLmodel, self.TimeSymbol )
        self.ParameterList = getParameter( aSBMLmodel, self.DerivedValueDic )
        self.ReactionList = getReaction( aSBMLmodel, aSBMLDocument, self.TimeSymbol )
        self.RuleList = getRule( aSBMLmodel, self.TimeSymbol )
        self.SpeciesList = getSpecies( aSBMLmodel, self.DerivedValueDic )
        self.UnitDefinitionList = getUnitDefinition( aSBMLmodel )

        self.setFunctionDefinitionToDictionary()
        

    # =========================================================

    def getPath( self, aCompartmentID ):
        return SBML_Base.getPath( self, aCompartmentID, self )

    # =========================================================

    def getSpeciesReferenceID( self, aSpeciesID ):
        return SBML_Base.getSpeciesReferenceID( self, aSpeciesID, self )

    # =========================================================

    def getVariableType( self, aName ):
        return SBML_Base.getVariableType( self, aName, self )

    # =========================================================
    
    def getID( self, anEntity ):
        return SBML_Base.getID( self, anEntity, self )

    # =========================================================

    def setFunctionDefinitionToDictionary( self ):

        if ( self.FunctionDefinitionList != [] ):

            for aFunctionDefinition in ( self.FunctionDefinitionList ):
                
                self.FunctionDefinition[ aFunctionDefinition.getID() ] = aFunctionDefinition[ 'Formula' ]
            

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

class SBML_Compartment( SBML_Base ):

    def __init__( self, aModel ):
        SBML_Base.__init__( self )
        self.Model = aModel

    # =========================================================

    def initialize( self, aCompartment ):

        self.__setSizeToDictionary( aCompartment )
        self.__setUnitToDictionary( aCompartment )

    # =========================================================

    def getPath( self, aCompartmentID ):
        return SBML_Base.getPath( self, aCompartmentID, self.Model )

    # =========================================================

    def getSpeciesReferenceID( self, aSpeciesID ):
        return SBML_Base.getSpeciesReferenceID( self, aSpeciesID, self.Model )

    # =========================================================

    def getVariableType( self, aName ):
        return SBML_Base.getVariableType( self, aName, self.Model )

    # =========================================================

    def getID( self, aSpecies ):
        return SBML_Base.getID( self, aSpecies, self.Model )

    # =========================================================
    
    def getCompartmentID( self, aCompartment ):
        
        if ( aCompartment[ 'Outside' ] == '' ):
            if ( self.Model.Level == 1 ):
                aSystemID = '/:' + aCompartment[ 'Name' ]
            elif ( self.Model.Level >= 2 ):
                aSystemID = '/:' + aCompartment[ 'Id' ]
            else:
                raise NameError,"Compartment Class needs a ['ID']"

        else:
            if( self.Model.Level == 1 ):
                aSystemID = self.getPath( aCompartment[ 'Outside' ] ) + ':'+ aCompartment[ 'Name' ]
            elif( self.Model.Level >= 2 ):
                aSystemID = self.getPath( aCompartment[ 'Outside' ] ) + ':'+ aCompartment[ 'Id' ]

        return 'System:' + aSystemID


    # =========================================================
    
    def __setSizeToDictionary( self, aCompartment ):

        if( self.Model.Level == 1 ):
            if( aCompartment[ 'Volume' ] != "Unknown" ):
                self.Model.CompartmentSize[ aCompartment[ 'Name' ] ] = aCompartment[ 'Volume' ]

            else:
                self.Model.CompartmentSize[ aCompartment[ 'Name' ] ] = self.__getOutsideSize( aCompartment[ 'Outside' ] )
                
        elif( self.Model.Level >= 2 ):
            if( aCompartment[ 'Size' ] != "Unknown" ):
                self.Model.CompartmentSize[ aCompartment[ 'Id' ] ] = aCompartment[ 'Size' ]

            else:
                self.Model.CompartmentSize[ aCompartment[ 'Id' ] ] = self.__getOutsideSize( aCompartment[ 'Outside' ] )


    # =========================================================
    
    def __setUnitToDictionary( self, aCompartment ):

        if( self.Model.Level == 1 ):
            aCompartmentID = aCompartment[ 'Name' ]

        elif( self.Model.Level >= 2 ):
            aCompartmentID = aCompartment[ 'Id' ]


        if( aCompartment[ 'Unit' ] != '' ):
            self.Model.CompartmentUnit[ aCompartmentID ] = aCompartment[ 'Unit' ]

        else:
            self.Model.CompartmentUnit[ aCompartmentID ] = self.__getOutsideUnit( aCompartment[ 'Outside' ] )


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

            return self.Model.CompartmentSize[ aCompartment[ 'Name' ] ]

        elif ( self.Model.Level >= 2 ):

            return self.Model.CompartmentSize[ aCompartment[ 'Id' ] ]


    # =========================================================    

    def getCompartmentUnit( self, aCompartment ):

        if ( self.Model.Level == 1 ):

            return self.Model.CompartmentUnit[ aCompartment[ 'Name' ] ]

        elif ( self.Model.Level >= 2 ):

            return self.Model.CompartmentUnit[ aCompartment[ 'Id' ] ]


    # =========================================================    


# --------------------------------
# Species Class
# --------------------------------

class SBML_Species( SBML_Base ):

    def __init__( self, aModel ):
        SBML_Base.__init__( self )
        self.Model = aModel
    

    # =========================================================

    def getPath( self, aCompartmentID ):
        return SBML_Base.getPath( self, aCompartmentID, self.Model )

    # =========================================================

    def getSpeciesReferenceID( self, aSpeciesID ):
        return SBML_Base.getSpeciesReferenceID( self, aSpeciesID, self.Model )

    # =========================================================

    def getVariableType( self, aName ):
        return SBML_Base.getVariableType( self, aName, self.Model )

    # =========================================================

    def getID( self, aSpecies ):
        return SBML_Base.getID( self, aSpecies, self.Model )

    # =========================================================
    
    def generateFullID( self, aSpecies ):

        aCompartmentID = aSpecies[ 'Compartment' ]

        if ( aCompartmentID == '' ):
            raise NameError, 'compartment property of Species must be defined'

        anEntityDic = {
            'Type'       : 'Variable',
            'Path'       : self.getPath( aCompartmentID ),
            'EntityName' : self.getID( aSpecies ) }

        return self.dic2FullID( anEntityDic )


    # =========================================================
    
    def getInitialValue( self, aSpecies ):

        if ( aSpecies[ 'InitialAmount' ] != 'Unknown' ): # initialAmount
            return { 
                'Property' : 'Value',
                'Value'    : float( aSpecies[ 'InitialAmount' ] ) }
        
        elif ( self.Model.Level >= 2 ):
            if ( aSpecies[ 'InitialConcentration' ] != 'Unknown' ): # initialConcentration

                if ( aSpecies[ 'Unit' ] == '' ):
                
                    # spatialSizeUnits and hasOnlySubstanceUnits should be checked
                
                    return { 
                        'Property' : 'NumberConc',
                        'Value'    : float( aSpecies[ 'InitialConcentration' ] ) }
                else:
                    raise ValueError, 'InitialAmount or InitialConcentration of Species [%s] must be defined.' % self.getID( aSpecies )

        raise ValueError, 'InitialAmount or InitialConcentration of Species [%s] must be defined.' % self.getID( aSpecies )

    # =========================================================

    def isConstant( self, aSpecies ):

        if ( self.Model.Level == 1 ):
            if ( aSpecies[ 'BoundaryCondition' ] == 1 ):
                return True
            else:
                return False
            
        elif ( self.Model.Level >= 2 ):
            if ( aSpecies[ 'Constant' ] == 1 ):
                return True
            else:
                return False

    # =========================================================


# --------------------------------
# Rule Class
# --------------------------------

class SBML_Rule( SBML_Base ):

    def __init__( self, aModel ):
        SBML_Base.__init__( self )
        self.Model = aModel


    # =========================================================

    def initialize( self ):

        self.VariableReferenceList = []
        self.VariableNumber = 0
##        self.ParameterNumber = 0


    # =========================================================

    def getPath( self, aCompartmentID ):
        return SBML_Base.getPath( self, aCompartmentID, self.Model )

    # =========================================================

    def getSpeciesReferenceID( self, aSpeciesID ):
        return SBML_Base.getSpeciesReferenceID( self, aSpeciesID, self.Model )

    # =========================================================

    def getVariableType( self, aName ):
        return SBML_Base.getVariableType( self, aName, self.Model )

    # =========================================================

    def getID( self, aSpecies ):
        return SBML_Base.getID( self, aSpecies, self.Model )

    # =========================================================
    
    def generateFullID( self, aRule ):
        if ( aRule[ 'Type' ] == libsbml.SBML_ALGEBRAIC_RULE ):
            anIDHeader = 'Algebraic'

        elif ( aRule[ 'Type' ] == libsbml.SBML_ASSIGNMENT_RULE            or
               aRule[ 'Type' ] == libsbml.SBML_SPECIES_CONCENTRATION_RULE or
               aRule[ 'Type' ] == libsbml.SBML_COMPARTMENT_VOLUME_RULE    or
               aRule[ 'Type' ] == libsbml.SBML_PARAMETER_RULE ):
            anIDHeader = 'Assignment'

        elif ( aRule[ 'Type' ] == libsbml.SBML_RATE_RULE ):
            anIDHeader = 'Rate'

        else:
            raise TypeError,\
                "Variable type must be Species, Parameter, or Compartment"

        anEntityDic = {
            'Type'       : 'Process',
            'Path'       : self.Model.EntityPath[ 'Rule' ],
            'EntityName' : '%s_%s' % ( anIDHeader, aRule[ 'Variable' ] ) }

        return self.dic2FullID( anEntityDic )


    # =========================================================

    def setSpeciesToVariableReference( self, aName, aStoichiometry='0' ):

        for aSpecies in self.Model.SpeciesList:

            if ( self.getID( aSpecies ) == aName ):
            
                for c in range( len( self.VariableReferenceList ) ):
                    aVariableReference = self.VariableReferenceList[ c ]
                    
                    if aVariableReference[ 1 ].split(':')[ 2 ] == aName:

                        aVariableReference[ 2 ] = str( int( aVariableReference[ 2 ] ) + int( aStoichiometry ))

                        return aVariableReference[ 0 ]

                aVariableList = []

                variableName = self.getID( aSpecies )
                aVariableList.append( variableName )
                self.VariableNumber = self.VariableNumber + 1

                aVariableID = self.getSpeciesReferenceID( aName )
                aVariableList.append( 'Variable:' + aVariableID )
                aVariableList.append( aStoichiometry )
                
                self.VariableReferenceList.append( aVariableList )

                return variableName

    # =========================================================

    def setParameterToVariableReference( self, aName, aStoichiometry='0' ):

        for aParameter in self.Model.ParameterList:

            if ( self.getID( aParameter ) == aName ):
                
                for c in range( len( self.VariableReferenceList ) ):
                    aVariableReference = self.VariableReferenceList[ c ]
                    
                    if aVariableReference[1].split(':')[2] == aName:

                        aVariableReference[ 2 ] = str( int( aVariableReference[ 2 ] ) + int( aStoichiometry ))

                        return aVariableReference[ 0 ]

                aParameterList = []
                variableName = aParameter[ 'Id' ]
                aParameterList.append( variableName )
##                self.ParameterNumber = self.ParameterNumber + 1
                aParameterList.append( ':'.join( [ 'Variable', self.EntityPath[ 'Parameter' ], aName ] ))
                aParameterList.append( aStoichiometry )
                self.VariableReferenceList.append( aParameterList )

                return variableName

    # =========================================================

    def setCompartmentToVariableReference( self, aName, aStoichiometry='0' ):

        for aCompartment in self.Model.CompartmentList:

            if ( self.getID( aCompartment ) ==aName ):
                
                for c in range( len( self.VariableReferenceList ) ):
                    aVariableReference = self.VariableReferenceList[ c ]
                    
                    if ( aVariableReference[1].split(':')[1] ==\
                       self.getPath( aName ) ) and\
                    ( aVariableReference[1].split(':')[2] == 'SIZE' ):

                        aVariableReference[ 2 ] = str( int( aVariableReference[ 2 ] ) + int( aStoichiometry ))

                        return aVariableReference[ 0 ]

                aCompartmentList = []
                aCompartmentList.append( aName )
                
                aCompartmentList.append(
                    'Variable:' + self.getPath( aName ) + ':SIZE' )
                
                aCompartmentList.append( aStoichiometry )
                self.VariableReferenceList.append( aCompartmentList )
                
                return aName
                            
    # =========================================================

    def __convertVariableName( self, anASTNode ):

        aNumChildren = anASTNode.getNumChildren()

        if ( aNumChildren > 0 ):
            for n in range( aNumChildren ):
                self.__convertVariableName( anASTNode.getChild( n ) )

        elif ( aNumChildren == 0 ):
            if ( anASTNode.isNumber() == 1 ):
                pass

            else:
                aName = anASTNode.getName()
                aType = self.getVariableType( aName )

                # Species
                if ( aType == libsbml.SBML_SPECIES ):

                    variableName = self.setSpeciesToVariableReference( aName )
                    if( variableName != '' ):

                        anASTNode.setType( libsbml.AST_NAME )
                        anASTNode.setName( '%s.NumberConc' % ( variableName ) )
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

    def convertFormula( self, aFormula ):
        preprocessedFormula = aFormula.replace( '<t>', self.Model.TimeSymbol )
        aASTRootNode = libsbml.parseFormula( preprocessedFormula )

        convertedAST = self.__convertVariableName( aASTRootNode )

        return postprocessMathString( libsbml.formulaToString( convertedAST ), self.Model.TimeSymbol )

    # =========================================================




# --------------------------------
# Reaction Class
# --------------------------------

class SBML_Reaction( SBML_Base ):

    def __init__( self, aModel ):
        SBML_Base.__init__( self )
        self.Model = aModel


    # =========================================================
    
    def initialize( self ):

##        self.SubstrateNumber = 0
        self.ProductNumber = 0
        self.ModifierNumber = 0
##        self.ParameterNumber = 0

        self.VariableReferenceList = []


    # =========================================================

    def getPath( self, aCompartmentID ):
        return SBML_Base.getPath( self, aCompartmentID, self.Model )

    # =========================================================

    def getSpeciesReferenceID( self, aSpeciesID ):
        return SBML_Base.getSpeciesReferenceID( self, aSpeciesID, self.Model )

    # =========================================================

    def getVariableType( self, aName ):
        return SBML_Base.getVariableType( self, aName, self.Model )

    # =========================================================

    def getID( self, aSpecies ):
        return SBML_Base.getID( self, aSpecies, self.Model )

    # =========================================================
    
    def generateFullID( self, aReaction ):

        if ( self.Model.Level == 1 ):
            if ( aReaction[1] != '' ):
                return 'Process:/:' + aReaction[1]
            else:
                raise NameError,"Reaction must set the Reaction name"
                
        elif ( self.Model.Level >= 2 ):
            if ( aReaction[ 'Id' ] != '' ):
                return 'Process:/:' + aReaction[ 'Id' ]
            else:
                raise NameError,"Reaction must set the Reaction ID"


    # =========================================================

    def setCompartmentToVariableReference( self, aName ):

        for aCompartment in self.Model.CompartmentList:
            if ( aCompartment[ 'Id' ] == aName or
                 aCompartment[ 'Name' ] == aName ):

                for aVariableReference in self.VariableReferenceList:
                    if( aVariableReference[1].split(':')[2] == 'SIZE' ):
                        aCurrentPath = ( aVariableReference[1].split(':')[1] )
                        aLastSlash = aCurrentPath.rindex( '/' )
                        variableName = aCurrentPath[aLastSlash+1:]
                        return aVariableReference[ 0 ]
                        ## return variableName
                                
                aCompartmentList = []
                aCompartmentList.append( aName )
                            
                aCompartmentList.append(
                    'Variable:' + self.getPath( aName ) + ':SIZE' )
                            
                aCompartmentList.append( '0' )
                self.VariableReferenceList.append( aCompartmentList )

                return aCompartmentList[0]

        return ''
    
    # =========================================================

    def __convertVariableName( self, anASTNode ):
        
        aNumChildren = anASTNode.getNumChildren()

        if ( aNumChildren > 0 ):

            #if ( anASTNode.isFunction() ):

                # Macro expand
                #if( self.Model.FunctionDefinition[ anASTNode.getName() ] != None ):
                #    self.macroExpand( anASTNode )

            for n in range( aNumChildren ):
                self.__convertVariableName( anASTNode.getChild( n ) )

            return anASTNode
        

        elif ( aNumChildren == 0 ):
            if ( anASTNode.isNumber() == 1 ):
                pass
            else:
                aName = anASTNode.getName()
                variableName = ''

                for aSpecies in self.Model.SpeciesList:
                    if ( aSpecies[ 'Id' ] == aName or aSpecies[ 'Name' ] == aName):

                        for aVariableReference in self.VariableReferenceList:
                            if aVariableReference[1].split(':')[2] == aName:
                                variableName =  aVariableReference[0]

                        if( self.Model.Level >= 2 and variableName == '' ):
                            raise NameError,"in libSBML :"+aName+" isn't defined in VariableReferenceList"

                        elif( self.Model.Level == 1 and variableName == '' ):

                            aModifierList = []
                            aModifierList.append(
                                'C' + str( self.ModifierNumber ) )
                            self.ModifierNumber = self.ModifierNumber + 1
                            
                            aModifierID = self.getSpeciesReferenceID( aName )
                            aModifierList.append( 'Variable:' + aModifierID )
                            aModifierList.append( '0' )
                            self.VariableReferenceList.append( aModifierList )
                            
                            variableName = aModifierList[0]

                        anASTNode.setType( libsbml.AST_NAME )
                        anASTNode.setName( '%s.NumberConc' % ( variableName ) )
                        
                        return anASTNode

##                if variableName == '':
                for aParameter in self.Model.ParameterList:
                    if ( aParameter[ 'Id' ] == aName or
                         aParameter[ 'Name' ] == aName ):

                        for aVariableReference in self.VariableReferenceList:
                            if aVariableReference[1].split(':')[2] == aName:
                                variableName = aVariableReference[0]

                        if( variableName == '' ):

                            aParameterList = []
                            aParameterList.append( aName )
                            
##                            self.ParameterNumber = self.ParameterNumber + 1

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
    
    def convertFormula( self, aFormula ):
        preprocessedFormula = aFormula.replace( '<t>', self.Model.TimeSymbol )
        aASTRootNode = libsbml.parseFormula( preprocessedFormula )

        convertedAST = self.__convertVariableName( aASTRootNode )

        return postprocessMathString( libsbml.formulaToString( convertedAST ), self.Model.TimeSymbol )

    # =========================================================

    def getVariableReference( self, anID ):

        for aVariableReference in self.VariableReferenceList:
            if( aVariableReference[0] == anID ):
                return aVariableReference
        
        return False


    # =========================================================

    def getStoichiometry( self, aSpeciesID, aStoichiometry ):

        if ( self.Model.Level == 1 ):
            for aSpecies in self.Model.SpeciesList:
                if ( aSpecies[ 'Name' ] == aSpeciesID ):
                    if( aSpecies[ 'BoundaryCondition' ] == 1 ):
                        return int( 0 )
                    else:
                        return int( aStoichiometry )

        elif ( self.Model.Level >= 2 ):
            for aSpecies in self.Model.SpeciesList:
                if ( aSpecies[ 'Id' ] == aSpeciesID ):
                    if( aSpecies[ 'Constant' ] == 1 ):
                        return int( 0 )
                    else:
                        return int( aStoichiometry )

        else:
           raise Exception,"Version"+str(self.Level)+" ????"


    # =========================================================

    def getChemicalEquation( self, aReaction ):

        # Left side (reactant terms)
        if ( len( aReaction[ 'Reactants' ] ) > 0 ):
            if ( aReaction[ 'Reactants' ][0][1] == 1.0 ):
                theLeftSide = "[%s]" % aReaction[ 'Reactants' ][0][0]
            else:
                theLeftSide = "%s x [%s]" % ( aReaction[ 'Reactants' ][0][1], aReaction[ 'Reactants' ][0][0] )

            for aSubstrate in aReaction[ 'Reactants' ][1:]:
                if ( aSubstrate[1] == 1.0 ):
                    theLeftSide += " + [%s]" % aSubstrate[0]
                else:
                    theLeftSide += " + %s x [%s]" % ( aSubstrate[1], aSubstrate[0] )

            theLeftSide += " "

        else:
            theLeftSide = ""

        # Right side (reactant terms)
        if ( len( aReaction[ 'Products' ] ) > 0 ):
            if ( aReaction[ 'Products' ][0][1] == 1.0 ):
                theRightSide = "[%s]" % aReaction[ 'Products' ][0][0]
            else:
                theRightSide = "%s x [%s]" % ( aReaction[ 'Products' ][0][1], aReaction[ 'Products' ][0][0] )

            for aProduct in aReaction[ 'Products' ][1:]:
                if ( aProduct[1] == 1.0 ):
                    theRightSide += " + [%s]" % aProduct[0]
                else:
                    theRightSide += " + %s x [%s]" % ( aProduct[1], aProduct[0] )

        else:
            theRightSide = ""

        if ( aReaction[ 'Reversible' ] == 0 ):
            theArrow = "->"
        else:
            theArrow = "<->"

        theEquation = "%s%s %s;" % ( theLeftSide, theArrow, theRightSide )

        # Effector
        if ( len( aReaction[ 'Modifiers' ] ) > 0 ):
            theEquation = "%s { %s };" % ( theEquation, ", ".join( aReaction[ 'Modifiers' ] ) )

        return theEquation


    # =========================================================





# --------------------------------
# Parameter Class
# --------------------------------

class SBML_Parameter( SBML_Base ):

    def __init__( self, aModel ):
        SBML_Base.__init__( self )
        self.Model = aModel


    # =========================================================

    def getPath( self, aCompartmentID ):
        return SBML_Base.getPath( self, aCompartmentID, self.Model )

    # =========================================================

    def getSpeciesReferenceID( self, aSpeciesID ):
        return SBML_Base.getSpeciesReferenceID( self, aSpeciesID, self.Model )

    # =========================================================

    def getVariableType( self, aName ):
        return SBML_Base.getVariableType( self, aName, self.Model )

    # =========================================================

    def getID( self, aSpecies ):
        return SBML_Base.getID( self, aSpecies, self.Model )

    # =========================================================

    def generateFullID( self, aParameter ):

        if ( self.Model.Level == 1 ):
            if ( aParameter[ 'Name' ] != '' ):
                return 'Variable:/SBMLParameter:' + aParameter[ 'Name' ]
            else:
                raise NameError, "Parameter must set the Parameter Name"
 
        elif ( self.Model.Level >= 2 ):
            if ( aParameter[ 'Id' ] != '' ):
                return 'Variable:/SBMLParameter:' + aParameter[ 'Id' ]
            else:
                raise NameError, "Parameter must set the Parameter ID"
 
        else:
            raise Exception,"Version"+str(self.Level)+" ????"
                 

   # =========================================================

    def getParameterValue( self, aParameter ):

        return aParameter[ 'Value' ]
   
##         if ( aParameter[ 'Unit' ] != '' and aParameter[ 'Value' ] != 0 ):

##             return self.convertUnit( aParameter[ 'Unit' ], aParameter[ 'Value' ] )
       
##         else:

##             return aParameter[ 'Value' ]
        

    # =========================================================



# --------------------------------
# Event Class
# --------------------------------

class SBML_Event( SBML_Base ):

    def __init__( self, aModel ):
        SBML_Base.__init__( self )
        self.Model = aModel
        self.EventNumber = 0


    # =========================================================

    def initialize( self ):

        self.VariableReferenceList = []
        self.VariableNumber = 0
##        self.ParameterNumber = 0
        self.EventNumber = self.EventNumber + 1


    # =========================================================

    def getPath( self, aCompartmentID ):
        return SBML_Base.getPath( self, aCompartmentID, self.Model )

    # =========================================================

    def getSpeciesReferenceID( self, aSpeciesID ):
        return SBML_Base.getSpeciesReferenceID( self, aSpeciesID, self.Model )

    # =========================================================

    def getVariableType( self, aName ):
        return SBML_Base.getVariableType( self, aName, self.Model )

    # =========================================================

    def getID( self, aSpecies ):
        return SBML_Base.getID( self, aSpecies, self.Model )

    # =========================================================

    def generateFullID( self, anEvent ):

        if( anEvent[ 'Id' ] != '' ):
            return 'Process:/SBMLEvent:' + anEvent[ 'Id' ]
        elif( anEvent[ 'Name' ] != '' ):
            return 'Process:/SBMLEvent:' + anEvent[ 'Name' ]
        else:
            anID = 'Process:/SBMLEvent:Event' + self.EventNumber
            self.EventNumber = self.EventNumber + 1
            return anID

    # =========================================================

    def getEventName( self, anEvent ):

        if( anEvent[ 'Id' ] != '' ):
            return anEvent[ 'Id' ]
        elif( anEvent[ 'Name' ] != '' ):
            return anEvent[ 'Name' ]
        else:
            return ""

    # =========================================================

    def setSpeciesToVariableReference( self, aName, aStoichiometry='0' ):

        for aSpecies in self.Model.SpeciesList:

            if ( ( self.Model.Level == 1 and aSpecies[ 'Name' ] == aName ) or
                 ( self.Model.Level >= 2 and aSpecies[ 'Id' ] == aName ) ):
            
                for c in range( len( self.VariableReferenceList ) ):
                    aVariableReference = self.VariableReferenceList[ c ]
                    
                    if aVariableReference[1].split(':')[2] == aName:

                        aVariableReference[ 2 ] = str( int( aVariableReference[ 2 ] ) + int( aStoichiometry ))

                        return aVariableReference[ 0 ]

                aVariableList = []

                variableName = aSpecies[ 'Id' ]
                aVariableList.append( variableName )
                self.VariableNumber = self.VariableNumber + 1

                aVariableID = self.getSpeciesReferenceID( aName )
                aVariableList.append( 'Variable:' + aVariableID )
                aVariableList.append( aStoichiometry )
                
                self.VariableReferenceList.append( aVariableList )

                return variableName

    # =========================================================

    def setParameterToVariableReference( self, aName, aStoichiometry='0' ):

        for aParameter in self.Model.ParameterList:

            if ( ( self.Model.Level == 1 and aParameter[ 'Name' ] == aName ) or
                 ( self.Model.Level >= 2 and aParameter[ 'Id' ] == aName ) ):
                
                for c in range( len( self.VariableReferenceList ) ):
                    aVariableReference = self.VariableReferenceList[ c ]
                    
                    if aVariableReference[1].split(':')[2] == aName:

                        aVariableReference[ 2 ] = str( int( aVariableReference[ 2 ] ) + int( aStoichiometry ))

                        return aVariableReference[ 0 ]

                aParameterList = []
                variableName = aParameter[ 'Id' ]
                aParameterList.append( variableName )
##                self.ParameterNumber = self.ParameterNumber + 1
                aParameterList.append( 'Variable:/SBMLParameter:' + aName )
                aParameterList.append( aStoichiometry )
                self.VariableReferenceList.append( aParameterList )

                return variableName

    # =========================================================

    def setCompartmentToVariableReference( self, aName, aStoichiometry='0' ):

        for aCompartment in self.Model.CompartmentList:

            if ( ( self.Model.Level == 1 and aCompartment[ 'Name' ] == aName ) or
                 ( self.Model.Level >= 2 and aCompartment[ 'Id' ] == aName ) ):
                
                for c in range( len( self.VariableReferenceList ) ):
                    aVariableReference = self.VariableReferenceList[ c ]
                    
                    if ( aVariableReference[1].split(':')[1] ==\
                       self.getPath( aName ) ) and\
                    ( aVariableReference[1].split(':')[2] == 'SIZE' ):

                        aVariableReference[ 2 ] = str( int( aVariableReference[ 2 ] ) + int( aStoichiometry ))

                        return aVariableReference[ 0 ]

                aCompartmentList = []
                aCompartmentList.append( aName )
                
                aCompartmentList.append(
                    'Variable:' + self.getPath( aName ) + ':SIZE' )
                
                aCompartmentList.append( aStoichiometry )
                self.VariableReferenceList.append( aCompartmentList )
                
                return aName

    # =========================================================

    def convertFormula( self, aFormula ):
        preprocessedFormula = aFormula.replace( '<t>', self.Model.TimeSymbol )
        aASTRootNode = libsbml.parseFormula( preprocessedFormula )

        convertedAST = self.__convertVariableName( aASTRootNode )

        return postprocessMathString( libsbml.formulaToString( convertedAST ), self.Model.TimeSymbol )

    # =========================================================

    def __convertVariableName( self, anASTNode ):

        aNumChildren = anASTNode.getNumChildren()

        if ( aNumChildren > 0 ):
            for n in range( aNumChildren ):
                self.__convertVariableName( anASTNode.getChild( n ) )

        elif ( aNumChildren == 0 ):
            if ( anASTNode.isNumber() == 1 ):
                pass

            else:
                aName = anASTNode.getName()
                
                # Time
                if ( aName == self.Model.TimeSymbol ):
                    anASTNode.setType( libsbml.AST_NAME_TIME )
                    return anASTNode

                else:
                    aType = self.getVariableType( aName )

                    # Species
                    if ( aType == libsbml.SBML_SPECIES ):

                        variableName = self.setSpeciesToVariableReference( aName )
                        if( variableName != '' ):

                            anASTNode.setType( libsbml.AST_NAME )
                            anASTNode.setName( '%s.NumberConc' % ( variableName ) )
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

