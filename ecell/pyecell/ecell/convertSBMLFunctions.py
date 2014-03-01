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

    def getPath( self, aCompartmentID, aModel ):

        if( aCompartmentID == 'default' ):
            return '/'

        for aCompartment in aModel.CompartmentList:
            if ( aCompartment[ aModel.keys[ 'ID' ] ] == aCompartmentID ):
                if aCompartment[ 'Outside' ] in ( '', 'default' ):
                    return '/' + aCompartmentID
                
                else:
                    return '%s/%s' % ( self.getPath( aCompartment[ 'Outside' ] ), aCompartmentID )


    # =========================================================

    def getSpeciesReferenceID( self, aSpeciesID, aModel ):

        for aSpecies in aModel.SpeciesList:
            if ( aSpecies[ aModel.keys[ 'ID' ] ] == aSpeciesID ):
                return '%s:%s' % ( SBML_Base.getPath( self, aSpecies[ 'Compartment' ], aModel ), aSpeciesID )


    # =========================================================

    def convertUnit( self, aValueUnit, aValue, aModel ):

        newValue = []
        for unitList in self.UnitDefinitionList:
            if ( unitList[ aModel.keys[ 'ID' ] ] == aValueUnit ):

                for anUnit in unitList[ 'Definition' ]:
                    aValue = aValue * self._getNewUnitValue( anUnit )

            newValue.append( aValue )


        if( newValue == [] ):
            return aValue
        else:
            return newValue[ 0 ]


    # =========================================================
    
    def getID( self, anEntity, aModel ):
        return anEntity[ aModel.keys[ 'ID' ] ]


    # =========================================================

    def dic2FullID( self, anEntityDic ):

        return "%(Type)s:%(Path)s:%(EntityName)s" % anEntityDic

    # =========================================================

    def getVariableType( self, aName, aModel ):

##        print "SBML_Base.getVariableType( %s )" % aName
        
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

    def getVariableFullID( self, anEntity, aModel ):
        aVariableType = SBML_Base.getVariableType( self, anEntity[ aModel.keys[ 'ID' ] ], aModel )
        if ( aVariableType == libsbml.SBML_SPECIES ):
            aPath = SBML_Base.getPath( self, anEntity[ 'Compartment' ], aModel )
            aName = anEntity[ aModel.keys[ 'ID' ] ]

        elif ( aVariableType == libsbml.SBML_PARAMETER ):
            aPath = self.EntityPath[ 'Parameter' ]
            aName = anEntity[ aModel.keys[ 'ID' ] ]

        elif ( aVariableType == libsbml.SBML_COMPARTMENT ):
            aPath = SBML_Base.getPath( self, anEntity[ aModel.keys[ 'ID' ] ], aModel )
            aName = 'SIZE'

        else:
            raise TypeError,\
                "Variable type must be Species, Parameter, or Compartment"

        anEntityDic = dict(
            Type       = 'Variable',
            Path       = aPath,
            EntityName = aName )

        return self.dic2FullID( anEntityDic )


    # =========================================================

    def updateVariableReferenceList( self, aModel, aVariableReferenceList, anID, aStoichiometry='0' ):

        aVariableType = self.getVariableType( anID )
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
            lambda anElement: self.getID( anElement ) == anID,
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
            aVariableReference.append( self.getVariableFullID( anEntity ) )
            aVariableReference.append( aStoichiometry )
            
            aVariableReferenceList.append( aVariableReference )

        return aVariableReference[ 0 ]


    # =========================================================

    def convertFormula( self, aFormula, aModel, aLocalParameterList = [] ):
        preprocessedFormula = aFormula.replace( '<t>', aModel.TimeSymbol )
        aASTRootNode = libsbml.parseFormula( preprocessedFormula )

        convertedAST = self._convertVariableName( aASTRootNode, aLocalParameterList )

        return postprocessMathString( libsbml.formulaToString( convertedAST ), aModel.TimeSymbol )

    # =========================================================

    def _convertVariableName( self, anASTNode, aLocalParameterList = [] ):

        aNumChildren = anASTNode.getNumChildren()

        if ( aNumChildren > 0 ):
            for n in range( aNumChildren ):
                self._convertVariableName( anASTNode.getChild( n ), aLocalParameterList )

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
                    variableName = self.updateVariableReferenceList( aName )
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

        if ( self.Level == 1 ):
           self.keys = dict( ID       = 'Name',
                             Size     = 'Volume',
                             Constant = 'BoundaryCondition' )
        else:
           self.keys = dict( ID       = 'Id',
                             Size     = 'Size',
                             Constant = 'Constant' )


    # =========================================================

    def hasEvent( self ):
        if ( self.EventList == [] ):
            return False
        else:
            return True

    def hasReaction( self ):
        if ( self.ReactionList == [] ):
            return False
        else:
            return True

    def hasSpecies( self ):
        if ( self.SpeciesList == [] ):
            return False
        else:
            return True

    # =========================================================

    def isApplicableVariableTimeStep( self ):

        for aReaction in self.ReactionList:
            if aReaction[ 'KineticLaw' ][ 'isDiscontinuous' ]:
                return False

        for aRule in self.RuleList:
            if aRule[ 'isDiscontinuous' ]:
                return False

        if self.hasEvent():
            return False

        return True

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

    def _getNewUnitValue( self, anUnit ):
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

        self._setSizeToDictionary( aCompartment )
        self._setUnitToDictionary( aCompartment )

    # =========================================================

    def getPath( self, aCompartmentID ):
        return super( SBML_Compartment, self ).getPath( aCompartmentID, self.Model )

    # =========================================================

    def getSpeciesReferenceID( self, aSpeciesID ):
        return super( SBML_Compartment, self ).getSpeciesReferenceID( aSpeciesID, self.Model )

    # =========================================================

    def getVariableType( self, aName ):
        return super( SBML_Compartment, self ).getVariableType( aName, self.Model )

    # =========================================================

    def getID( self, aSpecies ):
        return super( SBML_Compartment, self ).getID( aSpecies, self.Model )

    # =========================================================
    
    def getCompartmentID( self, aCompartment ):
        
        if ( aCompartment[ 'Outside' ] == '' ):
            return 'System:/:%s' % aCompartment[ self.Model.keys[ 'ID' ] ]

        else:
            return 'System:%s:%s' % ( self.getPath( aCompartment[ 'Outside' ] ), aCompartment[ self.Model.keys[ 'ID' ] ] )

    # =========================================================
    
    def _setSizeToDictionary( self, aCompartment ):

        if( aCompartment[ self.Model.keys[ 'Size' ] ] != "Unknown" ):
            self.Model.CompartmentSize[ aCompartment[ self.Model.keys[ 'ID' ] ] ] = aCompartment[ self.Model.keys[ 'Size' ] ]

        else:
            self.Model.CompartmentSize[ aCompartment[ self.Model.keys[ 'ID' ] ] ] = self._getOutsideSize( aCompartment[ 'Outside' ] )


    # =========================================================
    
    def _setUnitToDictionary( self, aCompartment ):

        aCompartmentID = aCompartment[ self.Model.keys[ 'ID' ] ]

        if( aCompartment[ 'Unit' ] != '' ):
            self.Model.CompartmentUnit[ aCompartmentID ] = aCompartment[ 'Unit' ]

        else:
            self.Model.CompartmentUnit[ aCompartmentID ] = self._getOutsideUnit( aCompartment[ 'Outside' ] )


    # =========================================================
    
    def _getOutsideSize( self, anOutsideCompartment ):
        
        if ( anOutsideCompartment == '' ):

            return float( 1 )

        else:
            return self.Model.CompartmentSize[ anOutsideCompartment ]


    # =========================================================
    
    def _getOutsideUnit( self, anOutsideCompartment ):

        if ( anOutsideCompartment == '' ):

            return ''

        else:
            return self.Model.CompartmentUnit[ anOutsideCompartment ]

    # =========================================================    

    def getCompartmentSize( self, aCompartment ):

        return self.Model.CompartmentSize[ aCompartment[ self.Model.keys[ 'ID' ] ] ]


    # =========================================================    

    def getCompartmentUnit( self, aCompartment ):

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

                if ( aSpecies[ 'Unit' ] in [ '', 'substance' ] ):
                
                    # spatialSizeUnits and hasOnlySubstanceUnits should be checked
                
                    return { 
                        'Property' : 'NumberConc',
                        'Value'    : float( aSpecies[ 'InitialConcentration' ] ) }
                else:
                    raise ValueError, 'InitialAmount or InitialConcentration of Species [%s] must be defined.' % self.getID( aSpecies )

        raise ValueError, 'InitialAmount or InitialConcentration of Species [%s] must be defined.' % self.getID( aSpecies )

    # =========================================================

    def isConstant( self, aSpecies ):

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
    
    def getSystemFullID( self ):

        anEntityDic = dict(
            Type       = 'System',
            Path       = self.EntityPath[ 'Root' ],
            EntityName = self.SystemName[ 'Rule' ] )

        return self.dic2FullID( anEntityDic )


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

    def getID( self, anEntity ):
        return SBML_Base.getID( self, anEntity, self.Model )

    # =========================================================

    def updateVariableReferenceList( self, anID, aStoichiometry='0' ):
        return SBML_Base.updateVariableReferenceList( self, self.Model, self.VariableReferenceList, anID, aStoichiometry )


    # =========================================================

    def getVariableFullID( self, anEntity ):
        return SBML_Base.getVariableFullID( self, anEntity, self.Model )


    # =========================================================
    
    def generateFullID( self, aRule ):
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

        return self.dic2FullID( anEntityDic )


    # =========================================================

    def convertFormula( self, aFormula, aLocalParameterList = [] ):
        return SBML_Base.convertFormula( self, aFormula, self.Model, aLocalParameterList )


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

    def updateVariableReferenceList( self, anID, aStoichiometry='0' ):
        return SBML_Base.updateVariableReferenceList( self, self.Model, self.VariableReferenceList, anID, aStoichiometry )


    # =========================================================

    def getVariableFullID( self, anEntity ):
        return SBML_Base.getVariableFullID( self, anEntity, self.Model )


    # =========================================================
    
    def generateFullID( self, aReaction ):

        if ( aReaction[ self.Model.keys[ 'ID' ] ] != '' ):
            return 'Process:/:' + aReaction[ self.Model.keys[ 'ID' ] ]
        else:
            raise NameError,"Reaction must set the Reaction name"


    # =========================================================

    def _convertVariableName( self, anASTNode, aLocalParameterList = [] ):
        
        aNumChildren = anASTNode.getNumChildren()

        if ( aNumChildren > 0 ):

            #if ( anASTNode.isFunction() ):

                # Macro expand
                #if( self.Model.FunctionDefinition[ anASTNode.getName() ] != None ):
                #    self.macroExpand( anASTNode )

            for n in range( aNumChildren ):
                self._convertVariableName( anASTNode.getChild( n ), aLocalParameterList )

            return anASTNode
        

        elif ( aNumChildren == 0 ):
            if ( anASTNode.isNumber() == 1 ):
                pass
            
            else:
            
                aName = anASTNode.getName()
                
##                print "Local Parameter: %s" % aLocalParameterList
                for aLocalParameter in aLocalParameterList:
                    if aLocalParameter[ self.Model.keys[ 'ID' ] ] == aName:
                        return anASTNode
                        
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
                variableName = self.updateVariableReferenceList( aName )
                if variableName != '':
                    anASTNode.setName( '%s.Value' % ( variableName ) )
                    return anASTNode
                
                return anASTNode

    # =========================================================
    
    def convertFormula( self, aFormula, aLocalParameterList = [] ):
        preprocessedFormula = aFormula.replace( '<t>', self.Model.TimeSymbol )
        aASTRootNode = libsbml.parseFormula( preprocessedFormula )

        convertedAST = self._convertVariableName( aASTRootNode, aLocalParameterList )

        return postprocessMathString( libsbml.formulaToString( convertedAST ), self.Model.TimeSymbol )

    # =========================================================

    def getVariableReference( self, anID ):

        for aVariableReference in self.VariableReferenceList:
            if( aVariableReference[0] == anID ):
                return aVariableReference
        
        return False


    # =========================================================

    def getStoichiometry( self, aSpeciesID, aStoichiometry ):

        for aSpecies in self.Model.SpeciesList:
            if ( aSpecies[ self.Model.keys[ 'ID' ] ] == aSpeciesID ):
                if( aSpecies[ self.Model.keys[ 'Constant' ] ] == 1 ):
                    return int( 0 )
                else:
                    return int( aStoichiometry )


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

        if ( aParameter[ self.Model.keys[ 'ID' ] ] != '' ):
            return 'Variable:/SBMLParameter:' + aParameter[ self.Model.keys[ 'ID' ] ]
        else:
            raise NameError, "Parameter must set the Parameter Name"

   # =========================================================

    def getParameterValue( self, aParameter ):

        return aParameter[ 'Value' ]
   
##         if ( aParameter[ 'Unit' ] != '' and aParameter[ 'Value' ] != 0 ):

##             return self.convertUnit( aParameter[ 'Unit' ], aParameter[ 'Value' ] )
       
##         else:

##             return aParameter[ 'Value' ]
        

    # =========================================================
    
    def getSystemFullID( self ):

        anEntityDic = dict(
            Type       = 'System',
            Path       = self.EntityPath[ 'Root' ],
            EntityName = self.SystemName[ 'Parameter' ] )

        return self.dic2FullID( anEntityDic )


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
    
    def getSystemFullID( self ):

        anEntityDic = dict(
            Type       = 'System',
            Path       = self.EntityPath[ 'Root' ],
            EntityName = self.SystemName[ 'Event' ] )

        return self.dic2FullID( anEntityDic )


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

    def updateVariableReferenceList( self, anID, aStoichiometry='0' ):
        return SBML_Base.updateVariableReferenceList( self, self.Model, self.VariableReferenceList, anID, aStoichiometry )


    # =========================================================

    def getVariableFullID( self, anEntity ):
        return SBML_Base.getVariableFullID( self, anEntity, self.Model )


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

    def convertFormula( self, aFormula, aLocalParameterList = [] ):
        return SBML_Base.convertFormula( self, aFormula, self.Model, aLocalParameterList )


    # =========================================================

