//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell MockProcess
//
//       Copyright (C) 1996-2008 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell MockProcess is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell MockProcess is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell MockProcess -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Koichi Takahashi
// modified by Moriyoshi Koizumi
//

#define BOOST_TEST_MODULE "MockProcess"

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <boost/test/test_case_template.hpp>
#include <boost/preprocessor/stringize.hpp>

#include "System.hpp"
#include "Variable.hpp"
#include "Process.hpp"
#include "PassiveStepper.hpp"
#include "Exceptions.hpp"
#include "dmtool/ModuleMaker.hpp"
#include "dmtool/DMObject.hpp"

#include <iostream>

using namespace libecs;

LIBECS_DM_CLASS( MockProcess, Process )
{
public:
    LIBECS_DM_OBJECT( MockProcess, Process )
    {
        INHERIT_PROPERTIES( Process );
    }

    virtual void fire() {}

    virtual void initialize()
    {
        Process::initialize();
    }    
};

LIBECS_DM_INIT_STATIC( MockProcess, Process );

BOOST_AUTO_TEST_CASE(testInstantiation)
{
    ModuleMaker< EcsObject > mmaker;
    DM_NEW_STATIC( &mmaker, EcsObject, MockProcess );
    
    MockProcess* proc = reinterpret_cast< MockProcess * >( mmaker.getModule( "MockProcess" ).getAllocator()() );
    BOOST_CHECK(proc);
    BOOST_CHECK_EQUAL("MockProcess", proc->getPropertyInterface().getClassName());
}

BOOST_AUTO_TEST_CASE(testGetMolarActivity)
{
    using namespace libecs;

    ModuleMaker< EcsObject > mmaker;
    DM_NEW_STATIC( &mmaker, EcsObject, MockProcess );
    DM_NEW_STATIC( &mmaker, EcsObject, Variable );
    DM_NEW_STATIC( &mmaker, EcsObject, System );

    System* sys = reinterpret_cast< System* >( mmaker.getModule( "System" ).getAllocator()() );

    MockProcess* proc = reinterpret_cast< MockProcess * >( mmaker.getModule( "MockProcess" ).getAllocator()() );
    proc->setActivity( N_A );
    BOOST_CHECK_EQUAL( N_A, proc->getActivity() );
    sys->registerProcess( proc );
    BOOST_CHECK_THROW( proc->getMolarActivity(), IllegalOperation );

    Variable* var = reinterpret_cast< Variable * >( mmaker.getModule( "Variable" ).getAllocator()() );
    var->setValue( 1.0 );
    var->setID( "SIZE" );
    sys->registerVariable( var );
    sys->configureSizeVariable();

    BOOST_CHECK_EQUAL( 1.0, proc->getMolarActivity() );
};

BOOST_AUTO_TEST_CASE(testGetStepper)
{
    using namespace libecs;

    ModuleMaker< EcsObject > mmaker;
    DM_NEW_STATIC( &mmaker, EcsObject, MockProcess );
    DM_NEW_STATIC( &mmaker, EcsObject, PassiveStepper );

    MockProcess* proc = reinterpret_cast< MockProcess * >( mmaker.getModule( "MockProcess" ).getAllocator()() );
    BOOST_CHECK_EQUAL( "", proc->getStepperID() );

    PassiveStepper* stepper = reinterpret_cast< PassiveStepper* >( mmaker.getModule( "PassiveStepper" ).getAllocator()() );
    stepper->setID( "STEPPER" );
    proc->setStepper( stepper );

    BOOST_CHECK_EQUAL( "STEPPER", proc->getStepperID() );
};

