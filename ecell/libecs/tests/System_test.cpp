//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2014 Keio University
//       Copyright (C) 2008-2014 RIKEN
//       Copyright (C) 2005-2009 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell System is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell System is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell System -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Koichi Takahashi
// modified by Moriyoshi Koizumi
//

#define BOOST_TEST_MODULE "System"

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <boost/test/test_case_template.hpp>
#include <boost/preprocessor/stringize.hpp>

#include "System.hpp"
#include "Variable.hpp"
#include "Exceptions.hpp"
#include "dmtool/ModuleMaker.hpp"
#include "dmtool/DMObject.hpp"

#include <iostream>

BOOST_AUTO_TEST_CASE(testInstantiation)
{
    using namespace libecs;

    ModuleMaker< EcsObject > mmaker;
    DM_NEW_STATIC( &mmaker, EcsObject, System );
    
    System* sys = reinterpret_cast< System * >( mmaker.getModule( "System" ).createInstance() );
    BOOST_CHECK(sys);
    BOOST_CHECK_EQUAL("System", sys->getPropertyInterface().getClassName());

    delete sys;
}

BOOST_AUTO_TEST_CASE(testGetSizeVariable)
{
    using namespace libecs;

    ModuleMaker< EcsObject > mmaker;
    DM_NEW_STATIC( &mmaker, EcsObject, System );
    DM_NEW_STATIC( &mmaker, EcsObject, Variable );
    
    System* sys = reinterpret_cast< System * >( mmaker.getModule( "System" ).createInstance() );
    BOOST_CHECK_THROW( sys->getSizeVariable(), IllegalOperation );

    Variable* var = reinterpret_cast< Variable * >( mmaker.getModule( "Variable" ).createInstance() );
    var->setValue( 123.0 );
    var->setID( "SIZE" );
    sys->registerEntity( var );
    sys->configureSizeVariable();

    BOOST_CHECK( sys->getSizeVariable() );
    BOOST_CHECK_EQUAL( 123.0, sys->getSize() );

    delete sys;
    delete var;
};
