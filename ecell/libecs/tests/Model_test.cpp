//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell LoggerBroker
//
//       Copyright (C) 1996-2009 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell LoggerBroker is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell LoggerBroker is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell LoggerBroker -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Koichi Takahashi
// modified by Moriyoshi Koizumi
//

#define BOOST_TEST_MODULE "Model"

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <boost/test/test_case_template.hpp>
#include <boost/preprocessor/stringize.hpp>

#include "Model.hpp"
#include "Variable.hpp"
#include "System.hpp"
#include "Process.hpp"
#include "Exceptions.hpp"
#include "dmtool/ModuleMaker.hpp"
#include "dmtool/DMObject.hpp"

#include <algorithm>

using namespace libecs;
#include "MockProcess.cpp"

BOOST_AUTO_TEST_CASE(testCreateEntity)
{
    ModuleMaker< EcsObject > mmaker;
    DM_NEW_STATIC( &mmaker, EcsObject, MockProcess );
    Model model( mmaker );
    model.setup();

    Variable* var( dynamic_cast< Variable* >(
            model.createEntity( "Variable", FullID( "Variable:/:test" ) ) ) );

    BOOST_CHECK( var != NULLPTR );
    BOOST_CHECK_EQUAL( var->getEntityType(), EntityType( "Variable" ) );
    BOOST_CHECK_EQUAL( var->getSystemPath().asString(), "/" );
    BOOST_CHECK_EQUAL( var->getID(), "test" );
    model.getEntity( FullID( "Variable:/:test" ) ); // assert no throw 

    Process* proc( dynamic_cast< Process* >(
            model.createEntity( "MockProcess", FullID( "Process:/:test" ) ) ) );

    BOOST_CHECK( proc != NULLPTR );
    BOOST_CHECK_EQUAL( proc->getEntityType(), EntityType( "Process" ) );
    BOOST_CHECK_EQUAL( proc->getSystemPath().asString(), "/" );
    BOOST_CHECK_EQUAL( proc->getID(), "test" );
    model.getEntity( FullID( "Process:/:test" ) ); // assert no throw 

    System* sys( dynamic_cast< System* >(
            model.createEntity( "System", FullID( "System:/:test" ) ) ) );

    BOOST_CHECK( sys != NULLPTR );
    BOOST_CHECK_EQUAL( sys->getEntityType(), EntityType( "System" ) );
    BOOST_CHECK_EQUAL( sys->getSystemPath().asString(), "/" );
    BOOST_CHECK_EQUAL( sys->getID(), "test" );
    model.getEntity( FullID( "System:/:test" ) ); // assert no throw 
}

BOOST_AUTO_TEST_CASE(testCreateStepper)
{
    ModuleMaker< EcsObject > mmaker;
    Model model( mmaker );
    model.setup();

    Stepper* stepper( model.createStepper( "PassiveStepper", "test" ) );

    BOOST_CHECK( stepper != NULLPTR );
    BOOST_CHECK_EQUAL( stepper->getID(), "test" );
    model.getStepper( "test" ); // assert no throw 
}

BOOST_AUTO_TEST_CASE(testDeleteStepper1)
{
    ModuleMaker< EcsObject > mmaker;
    Model model( mmaker );
    model.setup();

    Stepper* stepper( model.createStepper( "PassiveStepper", "test" ) );
    model.getStepper( "test" ); // assert no throw
    model.deleteStepper( "test" );
    BOOST_CHECK_THROW( model.getStepper( "test" ), NotFound );
}

BOOST_AUTO_TEST_CASE(testDeleteStepper2)
{
    ModuleMaker< EcsObject > mmaker;
    DM_NEW_STATIC( &mmaker, EcsObject, MockProcess );
    Model model( mmaker );
    model.setup();

    Stepper* stepper( model.createStepper( "PassiveStepper", "test" ) );
    model.getStepper( "test" ); // assert no throw

    model.getRootSystem()->setStepperID( "test" );
    model.createEntity( "MockProcess", FullID( "Process:/:test" ) );

    model.deleteStepper( "test" );
    BOOST_CHECK_THROW( model.getStepper( "test" ), NotFound );
}

BOOST_AUTO_TEST_CASE(testDeleteStepper3)
{
    ModuleMaker< EcsObject > mmaker;
    DM_NEW_STATIC( &mmaker, EcsObject, MockProcess );
    Model model( mmaker );
    model.setup();

    Stepper* stepper( model.createStepper( "PassiveStepper", "test" ) );
    model.getStepper( "test" ); // assert no throw

    model.getRootSystem()->setStepperID( "test" );
    model.createEntity( "MockProcess", FullID( "Process:/:test" ) );

    model.initialize();

    BOOST_CHECK_THROW( model.deleteStepper( "test" ), IllegalOperation );
}

BOOST_AUTO_TEST_CASE(testDeleteEntity1)
{
    ModuleMaker< EcsObject > mmaker;
    Model model( mmaker );
    model.setup();

    Stepper* stepper( model.createStepper( "PassiveStepper", "test" ) );
    model.getStepper( "test" ); // assert no throw

    FullID aFullID( "Variable:/:test" );
    BOOST_CHECK_THROW( model.getEntity( aFullID ), NotFound );
    model.createEntity( "Variable", aFullID );

    model.getRootSystem()->setStepperID( "test" );

    model.initialize();

    model.deleteEntity( aFullID );

    BOOST_CHECK_THROW( model.getEntity( aFullID ), NotFound );
    BOOST_CHECK_THROW( model.deleteEntity( aFullID ), NotFound );
}

BOOST_AUTO_TEST_CASE(testDeleteEntity2)
{
    ModuleMaker< EcsObject > mmaker;
    DM_NEW_STATIC( &mmaker, EcsObject, MockProcess );
    Model model( mmaker );
    model.setup();

    Stepper* stepper( model.createStepper( "PassiveStepper", "test" ) );
    model.getStepper( "test" ); // assert no throw

    FullID aVariableFullID( "Variable:/:test" );
    BOOST_CHECK_THROW( model.getEntity( aVariableFullID ), NotFound );
    Variable* aVariable( dynamic_cast< Variable* >( model.createEntity( "Variable", aVariableFullID ) ) );
    Process* aProcess1( dynamic_cast< Process* >( model.createEntity( "MockProcess", FullID( "Process:/:P1" ) ) ) );
    Process* aProcess2( dynamic_cast< Process* >( model.createEntity( "MockProcess", FullID( "Process:/:P2" ) ) ) );
    Integer aVarRefID1( aProcess1->registerVariableReference( "R", aVariable, 1 ) );
    Integer aVarRefID2( aProcess2->registerVariableReference( "R", aVariable, 1 ) );

    aProcess1->getVariableReference( "R" ); // no throw
    aProcess1->getVariableReference( aVarRefID1 ); // no throw
    aProcess2->getVariableReference( "R" ); // no throw
    aProcess2->getVariableReference( aVarRefID2 ); // no throw

    model.getRootSystem()->setStepperID( "test" );

    model.initialize();

    model.deleteEntity( aVariableFullID );

    BOOST_CHECK_THROW( model.getEntity( aVariableFullID ), NotFound );

    BOOST_CHECK_THROW( aProcess1->getVariableReference( "R" ), NotFound );
    BOOST_CHECK_THROW( aProcess1->getVariableReference( aVarRefID1 ), NotFound );
    BOOST_CHECK_THROW( aProcess2->getVariableReference( "R" ), NotFound );
    BOOST_CHECK_THROW( aProcess2->getVariableReference( aVarRefID2 ), NotFound );
}
