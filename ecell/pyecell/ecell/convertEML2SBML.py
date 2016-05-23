#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2016 Keio University
#       Copyright (C) 2008-2016 RIKEN
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

# Eml to Sbml converter

import sys
import os
import types
import getopt
import time
import sets

from ecell.ecssupport import *
from ecell.eml import *
from ecell.expressionparser import *
from ecell.SbmlFunctions import *

import libsbml
import numpy

def getCurrentCompartment( aSystemPath ):

    aLastSlash = aSystemPath.rindex( '/' )
    return aSystemPath[aLastSlash+1:]


# --------------------------------------------------------
# if Delay function "delay(,)" is defined in Expression,
# AST Node must be marked for recognizing it is "csymbol"
# --------------------------------------------------------

def setDelayType( anASTNode ):

    aNumChildren = anASTNode.getNumChildren()
    
    if ( aNumChildren == 2 ):
        
        if ( anASTNode.isFunction() == True and
             anASTNode.getName() == 'delay' ):
            
            anASTNode.setType( libsbml.AST_FUNCTION_DELAY )

        setDelayType( anASTNode.getLeftChild() )
        setDelayType( anASTNode.getRightChild() )

    elif ( aNumChildren == 1 ):
        setDelayType( anASTNode.getLeftChild() )

    return anASTNode


# -----------------------------------
# convert the number to the Mole Unit
# -----------------------------------

def convertToMoleUnit( aValue ):

    return float( aValue ) /  6.0221367e+23  # N_A 


# -------------------------------
# return the VariableReference ID
# -------------------------------

def getVariableReferenceId( aVariableReference, aCurrentSystem ):

    aFirstColon = aVariableReference.index( ':' )
    aLastColon = aVariableReference.rindex( ':' )

    # set Species Id to Reactant object
    if ( aVariableReference[aFirstColon+1:aLastColon] == '.' ):
        aSpeciesReferencePath = aCurrentSystem[1:].replace( '/', '__' )
        
    else:
        aSpeciesReferencePath = \
            aVariableReference[aFirstColon+2:aLastColon].replace( '/', '__' )

    aVariableID = aVariableReference[aLastColon+1:]

    if( aSpeciesReferencePath.count( '__' ) == 0 ):

        aSystem = aSpeciesReferencePath

    else:
        aLastUnderBar = aSpeciesReferencePath.rindex( '__' )
        aSystem = aSpeciesReferencePath[aLastUnderBar+2:]


    if ( aSpeciesReferencePath != '' ):   # Root system

        if( ( ( 'default__' + aVariableID ) in ID_Namespace ) == False ):
            if( aVariableID != 'SIZE' ):
                return aVariableReference[aLastColon+1:]
            else:
                return aSystem
        else:
            if( aVariableID != 'SIZE' ):
                return 'default__' + aVariableReference[aLastColon+1:]
            else:
                return 'default'

    elif( aSpeciesReferencePath == 'SBMLParameter' ):  # Parameter

        if( ( ( 'SBMLParameter__' + aVariableID ) in ID_Namespace ) == False ):
            return aVariableReference[aLastColon+1:]
        else:
            return 'SBMLParameter__' + aVariableReference[aLastColon+1:]

    else:    # other system        
        if( ( ( aSpeciesReferencePath + '__' + aVariableID ) in ID_Namespace )\
            == False ):

            if( aVariableID != 'SIZE' ):
                return aVariableID
            else:
                return aSystem
        else:
            if( aVariableID != 'SIZE' ):
                return aSpeciesReferencePath + '__' + aVariableID
            else:
                return aSpeciesReferencePath


#def getVariableID():


# --------------------------------------
#  set Expression Annotation for ecell3
# --------------------------------------

#def setExpressionAnnotation( aKineticLaw, anExpression ):

#   anExpressionAnnotation =\
#    '''<annotation xmlns:ecell="http://www.sbml.org/2001/ecell">
#              <ecell:Expression>
#              '''

#    anExpressionAnnotationEnd =\
#    '''
#              </ecell:Expression>
#          </annotation>'''
    
#    aKineticLaw.setAnnotation\
#    ( anExpressionAnnotation + anExpression + anExpressionAnnotationEnd )


# ------------------------------------------------------------
#  create Compartment, Species, Parameter and Reaction object
# ------------------------------------------------------------

def createEntity( anEml, aSBMLModel, aFullID, anOptional='' ):

    # create ID_Namespace set object for SBML Namespace
    global ID_Namespace
    ID_Namespace = sets.Set()


    aFullIDString = createFullIDString( aFullID )
    aType = ENTITYTYPE_STRING_LIST[ aFullID[ TYPE ] ]

    aClass = anEml.getEntityClass( aFullIDString )

    # ------------------------------------
    #  make the Species and the Parameter
    # ------------------------------------

    if ( aFullID[TYPE] == VARIABLE ):

        aPropertyNameList = anEml.getEntityPropertyList( aFullIDString )
        aCurrentCompartment = getCurrentCompartment( aFullID[1] )
        aCurrentCompartmentObj = aSBMLModel.getCompartment( aCurrentCompartment )

        if( aCurrentCompartment == "SBMLParameter" ):


            # ------------------------
            # create Parameter object
            # ------------------------
            aParameter = aSBMLModel.createParameter()

            if ( ( aFullID[2] in ID_Namespace ) == False ):
                aParameterID = aFullID[2]
            else:
                aParameterID = 'SBMLParamter__' + aFullID[2]


            # set Paramter ID to Id namespace
            ID_Namespace.add( aParameterID )
                    

            # set Parameter ID
            if( aSBMLLevel == '1' ):
                aParameter.setName( aParameterID )

            elif( aSBMLLevel == '2' ):
                aParameter.setId( aParameterID )


            # set Parameter Name, Value and Constant
            for aPropertyName in aPropertyNameList:
                
                aFullPN = aFullIDString + ':' + aPropertyName

                # set Parameter Name
                if ( aPropertyName == "Name" ):

                    if( aSBMLLevel == '1' ):
                        pass
                    
                    if( aSBMLLevel == '2' ):
                        aParameter.setName\
                        ( anEml.getEntityProperty( aFullPN ) )


                # set Parameter Value
                elif ( aPropertyName == "Value" ):

                    aParameter.setValue\
                    ( float( anEml.getEntityProperty( aFullPN )[0] ) )

                # set Constant 
                elif ( aPropertyName == "Fixed" ):
                    aParameter.setConstant\
                    ( int( float( anEml.getEntityProperty( aFullPN )[0] ) ) )

                else:
                    raise AttributeError, "Unrepresentable property `%s' in Parameter" % aPropertyName

        else:
            if( aFullID[2] != "SIZE" and aFullID[2] != "Dimensions" ):
                # create Species object
                aSpecies = aSBMLModel.createSpecies()

                # set Species ID
                if ( ( aFullID[2] in ID_Namespace ) == False ):
                    aSpeciesID = aFullID[2]
                else:
                    if ( aFullID[1][1:] != '' ):
                        aSpeciesID = aFullID[1][1:].replace( '/', '__' )\
                                     + '__' + aFullID[2]
                    else:
                        aSpeciesID = 'default__' + aFullID[2]

                ID_Namespace.add( aSpeciesID )

                if( aSBMLLevel == '1' ):
                    aSpecies.setName( aSpeciesID )
                if( aSBMLLevel == '2' ):
                    aSpecies.setId( aSpeciesID )


                # set Compartment of Species
                if( aCurrentCompartment == '' ):
                    aSpecies.setCompartment( 'default' )
                else:
                    aSpecies.setCompartment( aCurrentCompartment )
                
                # set Species Name, Value and Constant
                for aPropertyName in aPropertyNameList:

                    aFullPN = aFullIDString + ':' + aPropertyName

                    # set Species Name
                    if ( aPropertyName == "Name" ):

                        if( aSBMLLevel == '1' ):
                            pass

                        elif( aSBMLLevel == '2' ):
                            aSpecies.setName
                            ( anEml.getEntityProperty( aFullPN )[0] )

                    # set Species Value
                    elif ( aPropertyName == "Value" ):
                        
#                        aMolarValue = convertToMoleUnit(
#                            anEml.getEntityProperty( aFullPN )[0] )

                        aSpecies.setInitialAmount(
                            float( anEml.getEntityProperty( aFullPN )[0] )\
                            / 6.0221367e+23 )

                    # set Species Constant
                    elif ( aPropertyName == "Fixed" ):
                        
                        aSpecies.setConstant(
                            int( float( anEml.getEntityProperty( aFullPN )[0] ) ) )

                    # set Concentration by rule
                    elif ( aPropertyName == "MolarConc" ):
                        # XXX: units are just eventually correct here, because
                        # SBML falls back to mole and liter for substance and
                        # volume of the species if these are unspecified.
                        if aSBMLLevel == '1':
                            if aCurrentCompartmentObj != None:
                                compVol = float( \
                                        aCurrentCompartmentObj.getVolume() )
                            else:
                                compVol = 1.0
                            propValue = float( \
                                    anEml.getEntityProperty( aFullPN )[0] )
                            aSpecies.setInitialAmount( compVol * propValue )
                        else: # SBML lv.2
                            aSpecies.setInitialConcentration( \
                                float( anEml.getEntityProperty( aFullPN )[0] )\
                                )
                    else:
                        raise AttributeError, "Unrepresentable property `%s' in Species" % aPropertyName


    # -------------------------------------
    # make the Reaction and the Rule object
    # -------------------------------------
    
    elif ( aFullID[TYPE] == PROCESS ):

        aPropertyNameList = anEml.getEntityPropertyList( aFullIDString )

        aFirstColon = aFullIDString.index( ':' )
        aLastColon = aFullIDString.rindex( ':' )

        
        # ------------------
        #  make Rule object
        # ------------------

        if ( aFullIDString[aFirstColon+1:aLastColon] == '/SBMLRule' ):

            # get Process Class
            aProcessClass = anEml.getEntityClass( aFullIDString )
            aVariableReferenceList =\
            anEml.getEntityProperty( aFullIDString + ':VariableReferenceList' )
            aDelayFlag = False

            [ anExpression, aDelayFlag ] =\
            convertExpression(
                anEml.getEntityProperty( aFullIDString + ':Expression' )[0],
                aVariableReferenceList,
                aFullID[1],
                ID_Namespace )


            if( aProcessClass == 'ExpressionAlgebraicProcess' ):

                # create AlgebraicRule object
                anAlgebraicRule = aSBMLModel.createAlgebraicRule()

                # set AlgebraicRule Formula
                anAlgebraicRule.setFormula( anExpression )
                

            elif( aProcessClass == 'ExpressionAssignmentProcess' ):

                for aVariableReference in aVariableReferenceList:

                    if ( len( aVariableReference ) >= 3 ):

                        if ( aVariableReference[2] != '0' ): 

                            # create AssignmentRule object
                            anAssignmentRule =aSBMLModel.createAssignmentRule()

                            # set AssignmentRule Formula
                            anAssignmentRule.setFormula( aVariableReference[2] + '* ( ' + anExpression + ')' )
                            
                            aVariableID = getVariableReferenceId( aVariableReference[1], aFullID[1] )
                            anAssignmentRule.setVariable( aVariableID )

                        else:
                            pass

                    else:
                        pass
                        

            elif( aProcessClass == 'ExpressionFluxProcess' ):

                for aVariableReference in aVariableReferenceList:

                    if ( len( aVariableReference ) >= 3 ):
                        
                        if ( aVariableReference[2] != '0' ): 

                            # create AssignmentRule object
                            aRateRule = aSBMLModel.createRateRule()

                            # set AssignmentRule Formula
                            aRateRule.setFormula( aVariableReference[2] + '* ( ' + anExpression + ')' )

                            aVariableID = getVariableReferenceId( aVariableReference[1], aFullID[1] )
                            aRateRule.setVariable( aVariableID )

                        else:
                            pass

                    else:
                        pass


            else:
                raise TypeError, " The type of Process must be Algebraic, Assignment, Flux Processes"
            


        # ----------------------
        #  make Reaction object
        # ----------------------

        else:

            # create Parameter object
            aReaction = aSBMLModel.createReaction()

            # create KineticLaw Object
            aKineticLaw = aSBMLModel.createKineticLaw()

            # set Reaction ID
            if( aSBMLLevel == '1' ):
                aReaction.setName( aFullID[2] )
            if( aSBMLLevel == '2' ):
                aReaction.setId( aFullID[2] )


            for aPropertyName in aPropertyNameList:
                
                aFullPN = aFullIDString + ':' + aPropertyName

                # set Name property ( Name )
                if ( aPropertyName == "Name" ):
                
                    # set Reaction Name
                    if( aSBMLLevel == '1' ):
                        pass
                    
                    if( aSBMLLevel == '2' ):
                        aReaction.setName
                        ( anEml.getEntityProperty( aFullPN )[0] )


                # set Expression property ( KineticLaw Formula )
                elif ( aPropertyName == "Expression"):
                
                    # convert Expression of the ECELL format to
                    # SBML kineticLaw formula
                    anExpression = anEml.getEntityProperty( aFullPN )[0]
                    aVariableReferenceList =\
                    anEml.getEntityProperty( aFullIDString +\
                                             ':VariableReferenceList' )

#                    setExpressionAnnotation( aKineticLaw, anExpression )


                    aDelayFlag = False
                    [ anExpression, aDelayFlag ] =\
                      convertExpression( anExpression,
                                         aVariableReferenceList,
                                         aFullID[1],
                                         ID_Namespace )

                    # get Current System Id
                    for aVariableReference in aVariableReferenceList:

                        if( len( aVariableReference ) == 3 ):

                            if( int( float( aVariableReference[2]) ) != 0 ):
 
                                aFirstColon =\
                                aVariableReference[1].index( ':' )
                                aLastColon =\
                                aVariableReference[1].rindex( ':' )

                                if( aVariableReference[1][aFirstColon+1:aLastColon] == '.' ):

                                    aLastSlash = aFullID[1].rindex( '/' )

                                    CompartmentOfReaction=\
                                    aFullID[1][aLastSlash+1:]


                                else: 
                                    aLastSlash =\
                                    aVariableReference[1].rindex( '/' )

                                    CompartmentOfReaction=\
                                    aVariableReference[1][aLastSlash+1:aLastColon]
                            

                    if( CompartmentOfReaction == '' ):
                        
                        anExpression = '(' + anExpression + ')/default/N_A'

                    else:
                        
                        anExpression =\
                        '(' + anExpression + ')/' + CompartmentOfReaction + '/N_A'


                    # set KineticLaw Formula
                    if ( aDelayFlag == False ):
                        aKineticLaw.setFormula( anExpression )
                    else:
                        anASTNode = libsbml.parseFormula( anExpression )
                        anASTNode = setDelayType( anASTNode )

                        aKineticLaw.setMath( anASTNode )
                

                # set VariableReference property ( SpeciesReference )
                elif ( aPropertyName == "VariableReferenceList" ):

                    # make a flag. Because SBML model is defined
                    # both Product and Reactant. This flag is required
                    # in order to judge whether the Product and the
                    # Reactant are defined.

                    aReactantFlag = False
                    aProductFlag = False

                    for aVariableReference in anEml.getEntityProperty( aFullPN ):

                        if ( len( aVariableReference ) >= 3 ):

                            # --------------------------------
                            # add Reactants to Reaction object
                            # --------------------------------
                        
                            if ( float( aVariableReference[2] ) < 0 ):

                                # change the Reactant Flag
                                aReactantFlag = True
                            
                                # create Reactant object
                                aReactant = aSBMLModel.createReactant()

                                # set Species Id to Reactant object
                                aSpeciesReferenceId = getVariableReferenceId\
                                                      ( aVariableReference[1],\
                                                        aFullID[1] )

                                aReactant.setSpecies( aSpeciesReferenceId )


                                # set Stoichiometry 
                                aReactant.setStoichiometry(
                                    -( float( aVariableReference[2] ) ) )


                                # -------------------------------
                                # add Products to Reaction object
                                # -------------------------------
                        
                            elif ( float( aVariableReference[2] ) > 0 ):

                                # change the Product Flag
                                aProductFlag = True
                                
                                # create Product object
                                aProduct = aSBMLModel.createProduct()
                                
                                # set Species Id
                                aSpeciesReferenceId = getVariableReferenceId\
                                                      ( aVariableReference[1],\
                                                        aFullID[1] )

                                aProduct.setSpecies( aSpeciesReferenceId )
                                
                                # set Stoichiometry
                                aProduct.setStoichiometry(
                                    float( aVariableReference[2] ) )


                            # --------------------------------
                            # add Modifiers to Reaction object
                            # --------------------------------
                            
                            else:
                                # create Modifier object
                                aModifier = aSBMLModel.createModifier()
                                
                                # set Species Id to Modifier object
                                aVariableReferenceId = getVariableReferenceId( aVariableReference[1], aFullID[1] )

                                aModifier.setSpecies( aVariableReferenceId )

                            
                        # if there isn't the stoichiometry
                        elif ( len( aVariableReference ) == 2 ):

                            # create Modifier object
                            aModifier = aSBMLModel.createModifier()

                            # set Species Id to Modifier object
                            aVariableReferenceId = getVariableReferenceId( aVariableReference[1], aFullID[1] )

                            aModifier.setSpecies( aVariableReferenceId )



                    if ( aReactantFlag == False or aProductFlag == False ):

                        # set EmptySet Species, because if it didn't define,
                        # Reactant or Product can not be defined.
                    
                        if ( aReactantFlag == False ):

                            # create Reactant object
                            aReactant = aSBMLModel.createReactant()
                            
                            # set Species Id to Reactant object
                            aReactant.setSpecies( 'EmptySet' )

                            # set Stoichiometry 
                            aReactant.setStoichiometry( 0 )
                        

                        elif( aProductFlag == False ):

                            # create Product object
                            aProduct = aSBMLModel.createProduct()
                            
                            # set Species Id
                            aProduct.setSpecies( 'EmptySet' )
                            
                            # set Stoichiometry
                            aProduct.setStoichiometry( 0 )


                # These properties are not defined in SBML Lv2
                elif ( aPropertyName == "Priority" or
                       aPropertyName == "Activity" or
                       aPropertyName == "IsContinuous" or
                       aPropertyName == "StepperID" or
                       aPropertyName == "FireMethod" or
                       aPropertyName == "InitializeMethod" ):

                    pass

                else:
                
                    # create Parameter Object (Local)
                    aParameter = aSBMLModel.createKineticLawParameter()
                
                    # set Parameter ID
                    aParameter.setId( aPropertyName )

                    # set Parameter Value
                    aParameter.setValue(
                        float ( anEml.getEntityProperty( aFullPN )[0] ) )
                

            # add KineticLaw Object to Reaction Object
            aReaction.setKineticLaw( aKineticLaw )


    # --------------------
    # make the Compartment 
    # --------------------

    elif ( aFullID[TYPE] == SYSTEM ):

        if ( aFullID[2] != 'SBMLParameter' and aFullID[2] != 'SBMLRule' ):

            # create Compartment object
            aCompartment = aSBMLModel.createCompartment()
            aCompartmentID = ''

            # set ID ROOT System and Other System
            if( aFullID[2] == '' ):
                     aCompartmentID = 'default' # Root system
            else:
                if( ( aFullID[2] in ID_Namespace ) == False ):
                    aCompartmentID = aFullID[2]
                else:
                    if( aFullID[1][1:] == '' ):
                        aCompartmentID = 'default__' + aFullID[2]
                    else:
                        aCompartmentID = aFullID[1][1:].replace( '/', '__' )\
                                         + '__' + aFullID[2]

            ID_Namespace.add( aCompartmentID )
                
            if( aSBMLLevel == '1' ):
                aCompartment.setName( aCompartmentID )
            elif( aSBMLLevel == '2' ):
                aCompartment.setId( aCompartmentID )
                        

                    
            aSystemPath = aFullID[1] + aFullID[2] 

            for anID in anEml.getEntityList( 'Variable', aSystemPath ):

                # set Size and constant of Compartment
                if( anID == "SIZE" ):
                    
                    tmpPropertyList = anEml.getEntityPropertyList("Variable:" +
                                                                  aSystemPath +
                                                                  ":SIZE" )

                    for aProperty in tmpPropertyList:
                    
                        if ( aProperty == "Value" ):
                        
                            aFullPN = "Variable:" + aSystemPath + ':' +\
                                      anID + ':' + aProperty
                
                            aCompartment.setSize\
                            ( float( anEml.getEntityProperty( aFullPN )[0] ) )

                        elif ( aProperty == "Fixed" ):

                            aFullPN = "Variable:" + aSystemPath + ':' +\
                                      anID + ':' + aProperty

                            aCompartment.setConstant(
                                int( float( anEml.getEntityProperty( aFullPN )[0] ) ) )

                # set Dimensions of Compartment
                elif( anID == "Dimensions" ):

                    aFullPN = "Variable:" + aSystemPath + ':' + anID + ":Value"

                    aCompartment.setSpatialDimensions(
                        int( float( anEml.getEntityProperty( aFullPN )[0] ) ) )

            # set Outside element of Compartment
            if( aFullID[1] == '/' ):
                if( aFullID[2] != '' ):
                    aCompartment.setOutside( "default" )
                    
            else:
                aLastSlash = aFullID[1].rindex( '/' )

                aCompartment.setOutside(
                    getCurrentCompartment( aFullID[1][:aLastSlash] ) )




def createModel( anEml, aSBMLModel, aSystemPath='/' ):

    # set System
    if aSystemPath == '':
        aFullID = ( SYSTEM, '', '/' )
    else:
        aLastSlash = aSystemPath.rindex( '/' )
        aPath = aSystemPath[:aLastSlash+1]
        anID = aSystemPath[aLastSlash+1:]

        aFullID = ( SYSTEM, aPath, anID )

    createEntity( anEml, aSBMLModel, aFullID )

    # set Species
    for anID in anEml.getEntityList( 'Variable', aSystemPath ):

        aFullID = ( VARIABLE, aSystemPath, anID )
        createEntity( anEml, aSBMLModel, aFullID )

    # set Reaction
    for anID in anEml.getEntityList( 'Process', aSystemPath ):

        aFullID = ( PROCESS, aSystemPath, anID )
        createEntity( anEml, aSBMLModel, aFullID )

    # create SubSystem by iterating calling createModel
    for aSystem in anEml.getEntityList( 'System', aSystemPath ):
        aSubSystemPath = joinSystemPath( aSystemPath, aSystem )
        createModel( anEml, aSBMLModel, aSubSystemPath )



def setEssentialEntity( aSBMLModel ):

    # -------------------
    #  set N_A Parameter
    # -------------------

    isAbogadroNumber = False

    if( aSBMLLevel == '1' ):
        for aParameter in getParameter( aSBMLModel ):
            if( aParameter[1] == 'N_A' ):
                isAbogadroNumber = True

        if not( isAbogadroNumber ):
            # create Parameter object
            aParameter = aSBMLModel.createParameter()

            # set Parameter Name
            aParameter.setName( 'N_A' )
            
            # set Parameter Value
            aParameter.setValue( float( 6.0221367e+23 ) )
            
            # set Parameter Constant
            aParameter.setConstant( int( 1 ) )

    if( aSBMLLevel == '2' ):
        for aParameter in getParameter( aSBMLModel ):
            if( aParameter[0] == 'N_A' ):
                isAbogadroNumber = True

        if not( isAbogadroNumber ):
            # create Parameter object
            aParameter = aSBMLModel.createParameter()

            # set Parameter ID
            aParameter.setId( 'N_A' )
            
            # set Parameter Value
            aParameter.setValue( float( 6.0221367e+23 ) )
            
            # set Parameter Constant
            aParameter.setConstant( int( 1 ) )


    # ------------
    # set EmptySet
    # ------------

    isEmptySet = False

    if( aSBMLLevel == '1' ):
        for aSpecies in getSpecies( aSBMLModel ):
            if( aSpecies[1] == 'EmptySet' ):
                isEmptySet = True
    
        if not( isEmptySet ):
            # create Species object
            aSpecies = aSBMLModel.createSpecies()
    
            # set Species Name
            aSpecies.setName( 'EmptySet' )

            # set Species Compartment
            aSpecies.setCompartment( 'default' )
            
            # set Species Amount
            aSpecies.setInitialAmount( float( 0 ) )
            
            # set Species Constant
            aSpecies.setConstant( int( 1 ) )    


    if( aSBMLLevel == '2' ):
        for aSpecies in getSpecies( aSBMLModel ):

            if( aSpecies[0] == 'EmptySet' ):
                isEmptySet = True
    
        if not( isEmptySet ):
            # create Species object
            aSpecies = aSBMLModel.createSpecies()
    
            # set Species Id
            aSpecies.setId( 'EmptySet' )

            # set Species Compartment
            aSpecies.setCompartment( 'default' )
            
            # set Species Amount
            aSpecies.setInitialAmount( float( 0 ) )
            
            # set Species Constant
            aSpecies.setConstant( int( 1 ) )    
    

def convertToSBMLModel( anEml, aBaseName, aLevel, aVersion ):
    '''
    this function is called from ecell3-eml2sbml
    '''

    global aSBMLLevel
    aSBMLLevel = aLevel

    aSBMLDocument = libsbml.SBMLDocument()
    aSBMLDocument.setLevelAndVersion( int( aLevel ), int( aVersion ) )
    aSBMLModel = aSBMLDocument.createModel()

    aSBMLModel.setId( aBaseName )

    createModel( anEml, aSBMLModel )

    # set abogadro number and EmptySet to SBML model
    setEssentialEntity( aSBMLModel )

    return libsbml.writeSBMLToString( aSBMLDocument )
