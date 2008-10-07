//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2007 Keio University
//       Copyright (C) 2005-2007 The Molecular Sciences Institute
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
#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#define BOOST_TEST_MODULE "FullPN"

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <boost/test/test_case_template.hpp>
#include <boost/preprocessor/stringize.hpp>

#include "FullPN.hpp"
#include "Exceptions.hpp"

#include <iostream>

using libecs::FullPN;
using libecs::EntityType;
using libecs::SystemPath;
using libecs::BadFormat;

BOOST_AUTO_TEST_CASE(testFullPN)
{
    { 
        FullPN aFullPN( FullPN::parse( "       \t  \n  Variable:/A/B:S:prop   \t   \n" ) );
        BOOST_CHECK_EQUAL( aFullPN.asString(), "Variable:/A/B:S:prop" );
        BOOST_CHECK_EQUAL( aFullPN.getEntityType(), EntityType::VARIABLE ); 
        BOOST_CHECK( aFullPN.getSystemPath() == SystemPath::parse( "/A/B" ) ); 
        BOOST_CHECK_EQUAL( aFullPN.getID(), "S" ); 
        BOOST_CHECK_EQUAL( aFullPN.getPropertyName(), "prop" ); 
    }
    { 
        FullPN aFullPN( FullPN::parse( "Process::/:" ) );
        BOOST_CHECK_EQUAL( aFullPN.asString(), "Process::/:" );
        BOOST_CHECK_EQUAL( aFullPN.getEntityType(), EntityType::PROCESS ); 
        BOOST_CHECK( aFullPN.getSystemPath() == SystemPath::parse( "" ) ); 
        BOOST_CHECK_EQUAL( aFullPN.getID(), "/" ); 
        BOOST_CHECK_EQUAL( aFullPN.getPropertyName(), "" ); 
    }
}

BOOST_AUTO_TEST_CASE(testBadID)
{
    BOOST_CHECK_THROW( FullPN::parse( "Variable" ), BadFormat );
    BOOST_CHECK_THROW( FullPN::parse( "Variable:" ), BadFormat );
    BOOST_CHECK_THROW( FullPN::parse( "Variable::" ), BadFormat );
    BOOST_CHECK_THROW( FullPN::parse( "" ), BadFormat );
    BOOST_CHECK_THROW( FullPN::parse( ":" ), BadFormat );
    BOOST_CHECK_THROW( FullPN::parse( "::" ), BadFormat );
    BOOST_CHECK_THROW( FullPN::parse( ":::" ), BadFormat );
    BOOST_CHECK_THROW( FullPN::parse( ":/:A:prop" ), BadFormat );
    BOOST_CHECK_THROW( FullPN::parse( "::/:" ), BadFormat );
    BOOST_CHECK_THROW( FullPN::parse( "::A:prop" ), BadFormat );
    BOOST_CHECK_THROW( FullPN::parse( ":::prop" ), BadFormat );
    BOOST_CHECK_THROW( FullPN::parse( "UNKNOWN:/UNKNOWN:" ), BadFormat );
}

BOOST_AUTO_TEST_CASE(testAssignment)
{
    FullPN aFullPN( FullPN::parse( "Variable:/test:TEST:prop" ) );
    FullPN anAssignee( EntityType::NONE, SystemPath::parse(""), "", "" );

    anAssignee = aFullPN;
    BOOST_CHECK_EQUAL( anAssignee.asString(), "Variable:/test:TEST:prop" );
    BOOST_CHECK_EQUAL( anAssignee.getEntityType(), EntityType::VARIABLE ); 
    BOOST_CHECK( anAssignee.getSystemPath() == SystemPath::parse( "/test" ) ); 
    BOOST_CHECK_EQUAL( anAssignee.getEntityType(), EntityType::VARIABLE );
    BOOST_CHECK_EQUAL( anAssignee.getID(), "TEST" );
    BOOST_CHECK_EQUAL( anAssignee.getPropertyName(), "prop" );
    BOOST_CHECK_EQUAL( aFullPN.getEntityType(), EntityType::VARIABLE );
    BOOST_CHECK_EQUAL( aFullPN.getID(), "TEST" );
}
