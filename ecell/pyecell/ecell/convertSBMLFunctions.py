#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2015 Keio University
#       Copyright (C) 2008-2015 RIKEN
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

# Avogadro Number set
N_A = 6.0221367e+23





# --------------------------------
# Base Class
# --------------------------------

class SBML_Base( object ):


    def __init__( self ):

        self.SystemName = dict(
            Parameter = 'SBMLParameter',
            Rule      = 'SBMLRule',
            Reaction  = '',
            Event     = 'SBMLEvent' )

        self.EntityPath = dict( Root = '/' )

        for anEntity, aSystemName in self.SystemName.iteritems():
            self.EntityPath[ anEntity ] = '/' + aSystemName

    # =========================================================

    def get_path( self, aCompartmentID, aModel ):

        if( aCompartmentID == 'default' ):
            return '/'

        for aCompartment in aModel.CompartmentList:
            if ( aCompartment[ aModel.keys[ 'ID' ] ] == aCompartmentID ):
                if aCompartment[ 'Outside' ] in ( '', 'default' ):
                    return '/' + aCompartmentID
                
                else:
                    return '%s/%s' % ( self.get_path( aCompartment[ 'Outside' ] ), aCompartmentID )


    # =========================================================

    def get_SpeciesReference_ID( self, aSpeciesID, aModel ):

        for aSpecies in aModel.SpeciesList:
            if ( aSpecies[ aModel.keys[ 'ID' ] ] == aSpeciesID ):
                return '%s:%s' % ( SBML_Base.get_path( self, aSpecies[ 'Compartment' ], aModel ), aSpeciesID )


    # =========================================================

    def convert_unit( self, aValueUnit, aValue, aModel ):

        newValue = []
        for unitList in self.UnitDefinitionList:
            if ( unitList[ aModel.keys[ 'ID' ] ] == aValueUnit ):

                for anUnit in unitList[ 'Definition' ]:
                    aValue = aValue * self._get_new_unit_value( anUnit )

            newValue.append( aValue )


        if( newValue == [] ):
            return aValue
        else:
            return newValue[ 0 ]


    # =========================================================
    
    def get_ID( self, anEntity, aModel ):
        return anEntity[ aModel.keys[ 'ID' ] ]


    # =========================================================

    def convert_dic_to_FullID( self, anEntityDic ):

        return "%(Type)s:%(Path)s:%(EntityName)s" % anEntityDic

    # =========================================================

    def get_variable_type( self, aName, aModel ):

##        print "SBML_Base.get_variable_type( %s )" % aName
        
        IdKey = aModel.keys[ 'ID' ]

        for aSpecies in aModel.SpeciesList:
            if aSpecies[ IdKey ] == aName:
                return libsbml.SBML_SPECIES

        for aParameter in aModel.ParameterList:
            if aParameter[ IdKey ] == aName:
                return libsbml.SBML_PARAMETER

        for aCompartment in aModel.CompartmentList:
            if aCompartment[ IdKey ] == aName:
                return libsbml.SBML_COMPARTMENT

        raise TypeError, "Variable type must be Species, Parameter, or Compartment (got %s)" % aName


    # =========================================================

    def get_variable_FullID( self, anEntity, aModel ):
        aVariableType = SBML_Base.get_variable_type( self, anEntity[ aModel.keys[ 'ID' ] ], aModel )
        if ( aVariableType == libsbml.SBML_SPECIES ):
            aPath = SBML_Base.get_path( self, anEntity[ 'Compartment' ], aModel )
            aName = anEntity[ aModel.keys[ 'ID' ] ]

        elif ( aVariableType == libsbml.SBML_PARAMETER ):
            aPath = self.EntityPath[ 'Parameter' ]
            aName = anEntity[ aModel.keys[ 'ID' ] ]

        elif ( aVariableType == libsbml.SBML_COMPARTMENT ):
            aPath = SBML_Base.get_path( self, anEntity[ aModel.keys[ 'ID' ] ], aModel )
            aName = 'SIZE'

        else:
            raise TypeError,\
                "Variable type must be Species, Parameter, or Compartment"

        anEntityDic = dict(
            Type       = 'Variable',
            Path       = aPath,
            EntityName = aName )

        return self.convert_dic_to_FullID( anEntityDic )


    # =========================================================

    def add_Entity_to_VariableReferenceList( self, aModel, aVariableReferenceList, anID, aStoichiometry='0' ):

        aVariableType = self.get_variable_type( anID )
        if ( aVariableType == libsbml.SBML_SPECIES ):
            anEntityList = aModel.SpeciesList
        elif ( aVariableType == libsbml.SBML_PARAMETER ):
            anEntityList = aModel.ParameterList
        elif ( aVariableType == libsbml.SBML_COMPARTMENT ):
            anEntityList = aModel.CompartmentList
        else:
            raise TypeError,\
                "Variable type must be Species, Parameter, or Compartment"

        anEntity = filter( 
            lambda anElement: self.get_ID( anElement ) == anID,
            anEntityList )[ 0 ]
        
        aVariableReference = filter(
            lambda aVariableReference: aVariableReference[ 1 ].split(':')[ 2 ] == anID,
            aVariableReferenceList )

        if aVariableReference != []:
            aVariableReference = aVariableReference[ 0 ]
            aVariableReference[ 2 ] = str( int( aVariableReference[ 2 ] ) + int( aStoichiometry ))
            return aVariableReference[ 0 ]

        else:
            aVariableReference.append( anID )
            aVariableReference.append( self.get_variable_FullID( anEntity ) )
            aVariableReference.append( aStoichiometry )
            
            aVariableReferenceList.append( aVariableReference )

        return aVariableReference[ 0 ]


    # =========================================================

    def convert_SBML_Formula_to_ecell_Expression( self, formula, aModel, aLocalParameterList = [], aDenominator = 1.0 ):
        
        '''## =================================================
          formula: string or libsbml.ASTNode
        '''## =================================================
        
        if isinstance( formula, str ):
            if aDenominator != 1.0:
                formula = '( 1.0 / %s ) * ( %s )' % ( aDenominator, formula )
        
            preprocessedFormula = formula.replace( '<t>', self.Model.TimeSymbol )
            aASTRootNode = libsbml.parseFormula( preprocessedFormula )

        elif isinstance( formula, libsbml.ASTNode ):
           if aDenominator != 1.0:
                aASTRootNode = libsbml.parseFormula( '( 1.0 / %s ) * ( x )' % aDenominator )
                aASTRootNode.removeChild( 1 )
                aASTRootNode.addChild( formula.deepCopy() )
           else:
               aASTRootNode = formula

        else:
            raise Exception,"DEBUG : Formula must be str or libsbml.ASTNode instance."

##        dump_tree_construction_of_AST_node( aASTRootNode )

        aASTRootNode = preprocess_math_tree( aASTRootNode, aModel.TimeSymbol )

        convertedAST = self._convert_SBML_variable_to_ecell_Expression( aASTRootNode, aLocalParameterList )

        return postprocess_math_string( libsbml.formulaToString( convertedAST ), aModel.TimeSymbol )

    # =========================================================

    def _convert_SBML_variable_to_ecell_Expression( self, anASTNode, aLocalParameterList = [] ):

        aNumChildren = anASTNode.getNumChildren()

        if ( aNumChildren > 0 ):
            for n in range( aNumChildren ):
                self._convert_SBML_variable_to_ecell_Expression( anASTNode.getChild( n ), aLocalParameterList )

        elif ( aNumChildren == 0 ):
            if ( anASTNode.getType() == libsbml.AST_CONSTANT_E ):
                anASTNode.setType( libsbml.AST_REAL )
                anASTNode.setValue( math.e )
            
            elif ( anASTNode.getType() == libsbml.AST_CONSTANT_PI ):
                anASTNode.setType( libsbml.AST_REAL )
                anASTNode.setValue( math.pi )
            
            elif ( anASTNode.getType() == libsbml.AST_CONSTANT_FALSE ):
                anASTNode.setType( libsbml.AST_REAL )
                anASTNode.setValue( 0.0 )
            
            elif ( anASTNode.getType() == libsbml.AST_CONSTANT_TRUE ):
                anASTNode.setType( libsbml.AST_REAL )
                anASTNode.setValue( 1.0 )
            
            elif ( anASTNode.getType() == libsbml.AST_NAME_AVOGADRO ):
                anASTNode.setType( libsbml.AST_REAL )
                anASTNode.setValue( 6.0221367e+23 )
            
            elif ( anASTNode.isNumber() == 1 ):
                pass
            
            else:
                aName = anASTNode.getName()
                
                # Time
                if ( aName == self.Model.TimeSymbol ):
                    anASTNode.setType( libsbml.AST_NAME_TIME )
                    return anASTNode

                else:
                    variableName = self.add_Entity_to_VariableReferenceList( aName )
                    if( variableName != '' ):

                        anASTNode.setType( libsbml.AST_NAME )
                        anASTNode.setName( '%s.NumberConc' % ( variableName ) )
                        return anASTNode

        return anASTNode


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
        self.TimeSymbol = get_TimeSymbol( aSBMLmodel )

        self.CompartmentList = get_Compartment_list( aSBMLmodel )
        self.EventList = get_Event_list( aSBMLmodel, self.TimeSymbol )
        self.FunctionDefinitionList = get_FunctionDefinition_list( aSBMLmodel, self.TimeSymbol )
        self.ParameterList = get_Parameter_list( aSBMLmodel, self.DerivedValueDic )
        self.ReactionList = get_Reaction_list( aSBMLmodel, aSBMLDocument, self.TimeSymbol )
        self.RuleList = get_Rule_list( aSBMLmodel, self.TimeSymbol )
        self.SpeciesList = get_Species_list( aSBMLmodel, self.DerivedValueDic )
        self.UnitDefinitionList = get_UnitDefinition_list( aSBMLmodel )

        self.EntityDic = {
            libsbml.SBML_SPECIES     : self.SpeciesList,
            libsbml.SBML_PARAMETER   : self.ParameterList,
            libsbml.SBML_REACTION    : self.ReactionList,
            libsbml.SBML_COMPARTMENT : self.CompartmentList,
            libsbml.SBML_EVENT       : self.EventList
        }

        self.set_FunctionDefinition_to_dict()

    # =========================================================

        if ( self.Level == 1 ):
           self.keys = dict( ID       = 'Name',
                             Size     = 'Volume',
                             Constant = 'BoundaryCondition' )
        else:
           self.keys = dict( ID       = 'Id',
                             Size     = 'Size',
                             Constant = 'Constant' )


    # =========================================================

    def has_Event( self ):
        if ( self.EventList == [] ):
            return False
        else:
            return True

    def has_Reaction( self ):
        if ( self.ReactionList == [] ):
            return False
        else:
            return True

    def has_Species( self ):
        if ( self.SpeciesList == [] ):
            return False
        else:
            return True


    # =========================================================

    def get_Entity_by_ID( self, anID ):

        for ( anEntityType, anEntityList ) in self.EntityDic.items():
        
            hit = filter( 
                lambda anElement: anElement[ self.keys[ 'ID' ] ] == anID,
                anEntityList )
            
            if len( hit ):
                return ( anEntityType, hit[ 0 ] )

        return False


    # =========================================================

    def is_applicable_variable_time_step( self ):

        for aReaction in self.ReactionList:
            if aReaction[ 'KineticLaw' ][ 'isDiscontinuous' ]:
                return False

        for aRule in self.RuleList:
            if aRule[ 'isDiscontinuous' ]:
                return False

        if self.has_Event():
            return False

        return True

    # =========================================================

    def get_path( self, aCompartmentID ):
        return SBML_Base.get_path( self, aCompartmentID, self )

    # =========================================================

    def get_SpeciesReference_ID( self, aSpeciesID ):
        return SBML_Base.get_SpeciesReference_ID( self, aSpeciesID, self )

    # =========================================================

    def get_variable_type( self, aName ):
        return SBML_Base.get_variable_type( self, aName, self )

    # =========================================================
    
    def get_ID( self, anEntity ):
        return SBML_Base.get_ID( self, anEntity, self )

    # =========================================================

    def set_FunctionDefinition_to_dict( self ):

        if ( self.FunctionDefinitionList != [] ):

            for aFunctionDefinition in ( self.FunctionDefinitionList ):
                
                self.FunctionDefinition[ aFunctionDefinition.get_ID() ] = aFunctionDefinition[ 'Math' ]
            

    # =========================================================

    def _get_new_unit_value( self, anUnit ):
        return pow( pow( 10, anUnit[ 'Scale' ] ) * anUnit[ 'Multiplier' ], anUnit[ 'Exponent' ] ) + anUnit[ 'Offset' ]

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

        self._set_Compartment_size( aCompartment )
        self._set_Compartment_unit( aCompartment )

    # =========================================================

    def get_path( self, aCompartmentID ):
        return super( SBML_Compartment, self ).get_path( aCompartmentID, self.Model )

    # =========================================================

    def get_SpeciesReference_ID( self, aSpeciesID ):
        return super( SBML_Compartment, self ).get_SpeciesReference_ID( aSpeciesID, self.Model )

    # =========================================================

    def get_variable_type( self, aName ):
        return super( SBML_Compartment, self ).get_variable_type( aName, self.Model )

    # =========================================================

    def get_ID( self, aSpecies ):
        return super( SBML_Compartment, self ).get_ID( aSpecies, self.Model )

    # =========================================================
    
    def get_System_FullID( self, aCompartment ):
        
        anEntityDic = dict(
            Type       = 'System',
            EntityName = aCompartment[ self.Model.keys[ 'ID' ] ] )

        if ( aCompartment[ 'Outside' ] == '' ):
            anEntityDic[ 'Path' ] = '/'
        else:
            anEntityDic[ 'Path' ] = self.get_path( aCompartment[ 'Outside' ] )

        return self.convert_dic_to_FullID( anEntityDic )


    # =========================================================
    
    def _set_Compartment_size( self, aCompartment ):

        if( aCompartment[ self.Model.keys[ 'Size' ] ] != "Unknown" ):
            self.Model.CompartmentSize[ aCompartment[ self.Model.keys[ 'ID' ] ] ] = aCompartment[ self.Model.keys[ 'Size' ] ]

        else:
            self.Model.CompartmentSize[ aCompartment[ self.Model.keys[ 'ID' ] ] ] = self._get_outside_Compartment_size( aCompartment[ 'Outside' ] )


    # =========================================================
    
    def _set_Compartment_unit( self, aCompartment ):

        aCompartmentID = aCompartment[ self.Model.keys[ 'ID' ] ]

        if( aCompartment[ 'Unit' ] != '' ):
            self.Model.CompartmentUnit[ aCompartmentID ] = aCompartment[ 'Unit' ]

        else:
            self.Model.CompartmentUnit[ aCompartmentID ] = self._get_outside_compartment_unit( aCompartment[ 'Outside' ] )


    # =========================================================
    
    def _get_outside_Compartment_size( self, anOutsideCompartment ):
        
        if ( anOutsideCompartment == '' ):

            return float( 1 )

        else:
            return self.Model.CompartmentSize[ anOutsideCompartment ]


    # =========================================================
    
    def _get_outside_compartment_unit( self, anOutsideCompartment ):

        if ( anOutsideCompartment == '' ):

            return ''

        else:
            return self.Model.CompartmentUnit[ anOutsideCompartment ]

    # =========================================================    

    def get_Compartment_size( self, aCompartment ):

        return self.Model.CompartmentSize[ aCompartment[ self.Model.keys[ 'ID' ] ] ]


    # =========================================================    

    def get_Compartment_unit( self, aCompartment ):

        return self.Model.CompartmentUnit[ aCompartment[ self.Model.keys[ 'ID' ] ] ]


    # =========================================================    


# --------------------------------
# Species Class
# --------------------------------

class SBML_Species( SBML_Base ):

    def __init__( self, aModel ):
        SBML_Base.__init__( self )
        self.Model = aModel
    

    # =========================================================

    def get_path( self, aCompartmentID ):
        return SBML_Base.get_path( self, aCompartmentID, self.Model )

    # =========================================================

    def get_SpeciesReference_ID( self, aSpeciesID ):
        return SBML_Base.get_SpeciesReference_ID( self, aSpeciesID, self.Model )

    # =========================================================

    def get_variable_type( self, aName ):
        return SBML_Base.get_variable_type( self, aName, self.Model )

    # =========================================================

    def get_ID( self, aSpecies ):
        return SBML_Base.get_ID( self, aSpecies, self.Model )

    # =========================================================
    
    def generate_FullID_from_SBML_entity( self, aSpecies ):

        aCompartmentID = aSpecies[ 'Compartment' ]

        if ( aCompartmentID == '' ):
            raise NameError, 'compartment property of Species must be defined'

        anEntityDic = {
            'Type'       : 'Variable',
            'Path'       : self.get_path( aCompartmentID ),
            'EntityName' : self.get_ID( aSpecies ) }

        return self.convert_dic_to_FullID( anEntityDic )


    # =========================================================
    
    def get_initial_value( self, aSpecies ):

        if ( aSpecies[ 'InitialAmount' ] != 'Unknown' ): # initialAmount
            return { 
                'Property' : 'Value',
                'Value'    : float( aSpecies[ 'InitialAmount' ] ) }
        
        elif ( self.Model.Level >= 2 ):
            if ( aSpecies[ 'InitialConcentration' ] != 'Unknown' ): # initialConcentration

                if ( aSpecies[ 'Unit' ] in [ '', 'substance' ] ):
                
                    # spatialSizeUnits and hasOnlySubstanceUnits should be checked
                
                    return { 
                        'Property' : 'NumberConc',
                        'Value'    : float( aSpecies[ 'InitialConcentration' ] ) }
                else:
                    raise ValueError, 'InitialAmount or InitialConcentration of Species [%s] must be defined.' % self.get_ID( aSpecies )

        raise ValueError, 'InitialAmount or InitialConcentration of Species [%s] must be defined.' % self.get_ID( aSpecies )

    # =========================================================

    def is_constant( self, aSpecies ):

        if ( aSpecies[ self.Model.keys[ 'Constant' ] ] == 1 ):
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


    # =========================================================
    
    def get_System_FullID( self ):

        anEntityDic = dict(
            Type       = 'System',
            Path       = self.EntityPath[ 'Root' ],
            EntityName = self.SystemName[ 'Rule' ] )

        return self.convert_dic_to_FullID( anEntityDic )


    # =========================================================

    def get_path( self, aCompartmentID ):
        return SBML_Base.get_path( self, aCompartmentID, self.Model )

    # =========================================================

    def get_SpeciesReference_ID( self, aSpeciesID ):
        return SBML_Base.get_SpeciesReference_ID( self, aSpeciesID, self.Model )

    # =========================================================

    def get_variable_type( self, aName ):
        return SBML_Base.get_variable_type( self, aName, self.Model )

    # =========================================================

    def get_ID( self, anEntity ):
        return SBML_Base.get_ID( self, anEntity, self.Model )

    # =========================================================

    def add_Entity_to_VariableReferenceList( self, anID, aStoichiometry='0' ):
        return SBML_Base.add_Entity_to_VariableReferenceList( self, self.Model, self.VariableReferenceList, anID, aStoichiometry )


    # =========================================================

    def get_variable_FullID( self, anEntity ):
        return SBML_Base.get_variable_FullID( self, anEntity, self.Model )


    # =========================================================
    
    def generate_FullID_from_SBML_entity( self, aRule ):
        if ( aRule[ 'Type' ] == libsbml.SBML_ALGEBRAIC_RULE ):
            anIDHeader = 'Algebraic'

        elif aRule[ 'Type' ] in ( libsbml.SBML_ASSIGNMENT_RULE,
                                  libsbml.SBML_SPECIES_CONCENTRATION_RULE,
                                  libsbml.SBML_COMPARTMENT_VOLUME_RULE,
                                  libsbml.SBML_PARAMETER_RULE ):
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

        return self.convert_dic_to_FullID( anEntityDic )


    # =========================================================

    def convert_SBML_Formula_to_ecell_Expression( self, formula, aLocalParameterList = [], aDenominator = 1.0 ):
        return SBML_Base.convert_SBML_Formula_to_ecell_Expression( self, formula, self.Model, aLocalParameterList, aDenominator )


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

        self.ProductNumber = 0
        self.ModifierNumber = 0

        self.VariableReferenceList = []


    # =========================================================

    def get_path( self, aCompartmentID ):
        return SBML_Base.get_path( self, aCompartmentID, self.Model )

    # =========================================================

    def get_SpeciesReference_ID( self, aSpeciesID ):
        return SBML_Base.get_SpeciesReference_ID( self, aSpeciesID, self.Model )

    # =========================================================

    def get_variable_type( self, aName ):
        return SBML_Base.get_variable_type( self, aName, self.Model )

    # =========================================================

    def get_ID( self, aSpecies ):
        return SBML_Base.get_ID( self, aSpecies, self.Model )

    # =========================================================

    def add_Entity_to_VariableReferenceList( self, anID, aStoichiometry='0' ):
        return SBML_Base.add_Entity_to_VariableReferenceList( self, self.Model, self.VariableReferenceList, anID, aStoichiometry )


    # =========================================================

    def get_variable_FullID( self, anEntity ):
        return SBML_Base.get_variable_FullID( self, anEntity, self.Model )


    # =========================================================
    
    def generate_FullID_from_SBML_entity( self, aReaction ):

        if ( aReaction[ self.Model.keys[ 'ID' ] ] != '' ):
            return 'Process:/:' + aReaction[ self.Model.keys[ 'ID' ] ]
        else:
            raise NameError,"Reaction must set the Reaction name"


    # =========================================================

    def _convert_SBML_variable_to_ecell_Expression( self, anASTNode, aLocalParameterList = [] ):
        
        aNumChildren = anASTNode.getNumChildren()

        if ( aNumChildren > 0 ):

            #if ( anASTNode.isFunction() ):

                # Macro expand
                #if( self.Model.FunctionDefinition[ anASTNode.getName() ] != None ):
                #    self.macroExpand( anASTNode )

            for n in range( aNumChildren ):
                self._convert_SBML_variable_to_ecell_Expression( anASTNode.getChild( n ), aLocalParameterList )

            return anASTNode
        

        elif ( aNumChildren == 0 ):
            if ( anASTNode.getType() == libsbml.AST_CONSTANT_E ):
                anASTNode.setType( libsbml.AST_REAL )
                anASTNode.setValue( math.e )
            
            elif ( anASTNode.getType() == libsbml.AST_CONSTANT_PI ):
                anASTNode.setType( libsbml.AST_REAL )
                anASTNode.setValue( math.pi )
            
            elif ( anASTNode.getType() == libsbml.AST_CONSTANT_FALSE ):
                anASTNode.setType( libsbml.AST_REAL )
                anASTNode.setValue( 0.0 )
            
            elif ( anASTNode.getType() == libsbml.AST_CONSTANT_TRUE ):
                anASTNode.setType( libsbml.AST_REAL )
                anASTNode.setValue( 1.0 )
            
            elif ( anASTNode.getType() == libsbml.AST_NAME_AVOGADRO ):
                anASTNode.setType( libsbml.AST_REAL )
                anASTNode.setValue( 6.0221367e+23 )
            
            elif ( anASTNode.isNumber() == 1 ):
                pass
                
            else:
                aName = anASTNode.getName()
                
##                print "Local Parameter: %s" % aLocalParameterList
                for aLocalParameter in aLocalParameterList:
                    if aLocalParameter[ self.Model.keys[ 'ID' ] ] == aName:
                        return anASTNode
                        
                variableName = ''

                for aSpecies in self.Model.SpeciesList:
                    if aSpecies[ self.Model.keys[ 'ID' ] ] == aName:

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
                            
                            aModifierID = self.get_SpeciesReference_ID( aName )
                            aModifierList.append( 'Variable:' + aModifierID )
                            aModifierList.append( '0' )
                            self.VariableReferenceList.append( aModifierList )
                            
                            variableName = aModifierList[0]

                        anASTNode.setType( libsbml.AST_NAME )
                        anASTNode.setName( '%s.NumberConc' % ( variableName ) )
                        
                        return anASTNode

##                if variableName == '':
                for aParameter in self.Model.ParameterList:
                    if aParameter[ self.Model.keys[ 'ID' ] ] == aName:

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
                variableName = self.add_Entity_to_VariableReferenceList( aName )
                if variableName != '':
                    anASTNode.setName( '%s.Value' % ( variableName ) )
                    return anASTNode
                
                return anASTNode

    # =========================================================

    def convert_SBML_Formula_to_ecell_Expression( self, formula, aLocalParameterList = [], aDenominator = 1.0 ):
        return SBML_Base.convert_SBML_Formula_to_ecell_Expression( self, formula, self.Model, aLocalParameterList, aDenominator )


    # =========================================================

    def get_VariableReference( self, anID ):

        for aVariableReference in self.VariableReferenceList:
            if( aVariableReference[0] == anID ):
                return aVariableReference
        
        return False


    # =========================================================

    def get_chemical_equation( self, aReaction ):

        # Left side (reactant terms)
        if ( len( aReaction[ 'Reactants' ] ) > 0 ):
            if ( aReaction[ 'Reactants' ][0][ 'Stoichiometry' ] == 1.0 ):
                theLeftSide = "[%s]" % aReaction[ 'Reactants' ][0][ self.Model.keys[ 'ID' ] ]
            else:
                theLeftSide = "%s x [%s]" % ( int( aReaction[ 'Reactants' ][0][ 'Stoichiometry' ] * aReaction[ 'CommonDemoninator' ] ), aReaction[ 'Reactants' ][0][ self.Model.keys[ 'ID' ] ] )

            for aSubstrate in aReaction[ 'Reactants' ][1:]:
                if ( aSubstrate[ 'Stoichiometry' ] == 1.0 ):
                    theLeftSide += " + [%s]" % aSubstrate[ self.Model.keys[ 'ID' ] ]
                else:
                    theLeftSide += " + %s x [%s]" % ( int( aSubstrate[ 'Stoichiometry' ] * aReaction[ 'CommonDemoninator' ] ), aSubstrate[ self.Model.keys[ 'ID' ] ] )

            theLeftSide += " "

        else:
            theLeftSide = ""

        # Right side (reactant terms)
        if ( len( aReaction[ 'Products' ] ) > 0 ):
            if ( aReaction[ 'Products' ][0][ 'Stoichiometry' ] == 1.0 ):
                theRightSide = "[%s]" % aReaction[ 'Products' ][0][ self.Model.keys[ 'ID' ] ]
            else:
                theRightSide = "%s x [%s]" % ( int( aReaction[ 'Products' ][0][ 'Stoichiometry' ] * aReaction[ 'CommonDemoninator' ] ), aReaction[ 'Products' ][0][ self.Model.keys[ 'ID' ] ] )

            for aProduct in aReaction[ 'Products' ][1:]:
                if ( aProduct[ 'Stoichiometry' ] == 1.0 ):
                    theRightSide += " + [%s]" % aProduct[ self.Model.keys[ 'ID' ] ]
                else:
                    theRightSide += " + %s x [%s]" % ( int( aProduct[ 'Stoichiometry' ] * aReaction[ 'CommonDemoninator' ] ), aProduct[ self.Model.keys[ 'ID' ] ] )

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

    def get_path( self, aCompartmentID ):
        return SBML_Base.get_path( self, aCompartmentID, self.Model )

    # =========================================================

    def get_SpeciesReference_ID( self, aSpeciesID ):
        return SBML_Base.get_SpeciesReference_ID( self, aSpeciesID, self.Model )

    # =========================================================

    def get_variable_type( self, aName ):
        return SBML_Base.get_variable_type( self, aName, self.Model )

    # =========================================================

    def get_ID( self, aSpecies ):
        return SBML_Base.get_ID( self, aSpecies, self.Model )

    # =========================================================

    def generate_FullID_from_SBML_entity( self, aParameter ):

        if ( aParameter[ self.Model.keys[ 'ID' ] ] != '' ):
            return 'Variable:/SBMLParameter:' + aParameter[ self.Model.keys[ 'ID' ] ]
        else:
            raise NameError, "Parameter must set the Parameter Name"

    # =========================================================
    
    def get_System_FullID( self ):

        anEntityDic = dict(
            Type       = 'System',
            Path       = self.EntityPath[ 'Root' ],
            EntityName = self.SystemName[ 'Parameter' ] )

        return self.convert_dic_to_FullID( anEntityDic )


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
    
    def get_System_FullID( self ):

        anEntityDic = dict(
            Type       = 'System',
            Path       = self.EntityPath[ 'Root' ],
            EntityName = self.SystemName[ 'Event' ] )

        return self.convert_dic_to_FullID( anEntityDic )


    # =========================================================

    def get_path( self, aCompartmentID ):
        return SBML_Base.get_path( self, aCompartmentID, self.Model )

    # =========================================================

    def get_SpeciesReference_ID( self, aSpeciesID ):
        return SBML_Base.get_SpeciesReference_ID( self, aSpeciesID, self.Model )

    # =========================================================

    def get_variable_type( self, aName ):
        return SBML_Base.get_variable_type( self, aName, self.Model )

    # =========================================================

    def get_ID( self, aSpecies ):
        return SBML_Base.get_ID( self, aSpecies, self.Model )

    # =========================================================

    def add_Entity_to_VariableReferenceList( self, anID, aStoichiometry='0' ):
        return SBML_Base.add_Entity_to_VariableReferenceList( self, self.Model, self.VariableReferenceList, anID, aStoichiometry )


    # =========================================================

    def get_variable_FullID( self, anEntity ):
        return SBML_Base.get_variable_FullID( self, anEntity, self.Model )


    # =========================================================

    def generate_FullID_from_SBML_entity( self, anEvent ):

        if( anEvent[ 'Id' ] != '' ):
            return 'Process:/SBMLEvent:' + anEvent[ 'Id' ]
        elif( anEvent[ 'Name' ] != '' ):
            return 'Process:/SBMLEvent:' + anEvent[ 'Name' ]
        else:
            anID = 'Process:/SBMLEvent:Event' + self.EventNumber
            self.EventNumber = self.EventNumber + 1
            return anID

    # =========================================================

    def get_Event_Name( self, anEvent ):

        if( anEvent[ 'Id' ] != '' ):
            return anEvent[ 'Id' ]
        elif( anEvent[ 'Name' ] != '' ):
            return anEvent[ 'Name' ]
        else:
            return ""

    # =========================================================

    def convert_SBML_Formula_to_ecell_Expression( self, formula, aLocalParameterList = [], aDenominator = 1.0 ):
        return SBML_Base.convert_SBML_Formula_to_ecell_Expression( self, formula, self.Model, aLocalParameterList, aDenominator )


    # =========================================================

