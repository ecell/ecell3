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

#define BOOST_TEST_MODULE "FullID"

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <boost/test/test_case_template.hpp>
#include <boost/preprocessor/stringize.hpp>

#include "FullID.hpp"
#include "Exceptions.hpp"

#include <iostream>

using libecs::FullID;
using libecs::EntityType;
using libecs::SystemPath;
using libecs::BadFormat;

BOOST_AUTO_TEST_CASE(testConstruction)
{
    {
        SystemPath aSystemPath( "/A/B/C/" );
        BOOST_CHECK_EQUAL( aSystemPath.getString(),
                           "/A/B/C/" );
    }
    {
        SystemPath aSystemPath( "." );
        BOOST_CHECK_EQUAL( aSystemPath.getString(), "." );
    }
    {
        SystemPath aSystemPath( ".." );
        BOOST_CHECK_EQUAL( aSystemPath.getString(), ".." );
    }
    {
        SystemPath aSystemPath( "" );
        BOOST_CHECK_EQUAL( aSystemPath.getString(), "" );
    }
    {
        SystemPath aSystemPath(
            "   \t  /A/BB/CCC/../DDDD/EEEEEE    \t \n  " );
        BOOST_CHECK_EQUAL( aSystemPath.getString(),
                           "/A/BB/CCC/../DDDD/EEEEEE" );
    }
}

BOOST_AUTO_TEST_CASE(testCopyConstruction)
{
    SystemPath aSystemPath( "   \t  /A/BB/CCC/../DDDD/EEEEEE    \t \n  " );
    SystemPath aSystemPath2( aSystemPath );
    BOOST_CHECK_EQUAL( aSystemPath2.getString(),
                       "/A/BB/CCC/../DDDD/EEEEEE" );
}

BOOST_AUTO_TEST_CASE(testPopFront)
{
    {
        SystemPath aSystemPath( "/A/BB/CCC/../DDDD/EEEEEE" );
        aSystemPath.pop_front();
        BOOST_CHECK_EQUAL( aSystemPath.getString(),
                           "A/BB/CCC/../DDDD/EEEEEE" );
        aSystemPath.pop_front();
        BOOST_CHECK_EQUAL( aSystemPath.getString(),
                           "BB/CCC/../DDDD/EEEEEE" );
        aSystemPath.pop_front();
        BOOST_CHECK_EQUAL( aSystemPath.getString(),
                           "CCC/../DDDD/EEEEEE" );
        aSystemPath.pop_front();
        BOOST_CHECK_EQUAL( aSystemPath.getString(),
                           "../DDDD/EEEEEE" );
        aSystemPath.pop_front();
        BOOST_CHECK_EQUAL( aSystemPath.getString(),
                           "DDDD/EEEEEE" );
        aSystemPath.pop_front();
        BOOST_CHECK_EQUAL( aSystemPath.getString(),
                           "EEEEEE" );
        aSystemPath.pop_front();
        BOOST_CHECK_EQUAL( aSystemPath.getString(),
                           "" );
    }

    {
        SystemPath aSystemPath( "/" );
        aSystemPath.pop_front();
        BOOST_CHECK_EQUAL( aSystemPath.getString(), "" );
    }
}

BOOST_AUTO_TEST_CASE(testPopBack)
{
    {
        SystemPath aSystemPath( "/A/BB/CCC/../DDDD/EEEEEE" );
        aSystemPath.pop_back();
        BOOST_CHECK_EQUAL( aSystemPath.getString(),
                           "/A/BB/CCC/../DDDD" );
        aSystemPath.pop_back();
        BOOST_CHECK_EQUAL( aSystemPath.getString(),
                           "/A/BB/CCC/.." );
        aSystemPath.pop_back();
        BOOST_CHECK_EQUAL( aSystemPath.getString(),
                           "/A/BB/CCC" );
        aSystemPath.pop_back();
        BOOST_CHECK_EQUAL( aSystemPath.getString(),
                           "/A/BB" );
        aSystemPath.pop_back();
        BOOST_CHECK_EQUAL( aSystemPath.getString(),
                           "/A" );
    }

    {
        SystemPath aSystemPath( "/" );
        aSystemPath.pop_back();
        BOOST_CHECK_EQUAL( aSystemPath.getString(), "" );
    }
}

BOOST_AUTO_TEST_CASE(testIsAbsolute)
{
    BOOST_CHECK( SystemPath( "/A/BB/CCC/../DDDD/EEEEEE" ).isAbsolute() );
    BOOST_CHECK( !SystemPath( "A/.." ).isAbsolute() );
    BOOST_CHECK( SystemPath( "/" ).isAbsolute() );
    BOOST_CHECK( SystemPath( "" ).isAbsolute() ); // XXX: is this OK?
    BOOST_CHECK( !SystemPath( ".." ).isAbsolute() );
    BOOST_CHECK( SystemPath( "/.." ).isAbsolute() );
}

BOOST_AUTO_TEST_CASE(testFullID)
{
    { 
        FullID aFullID( "       \t  \n  Variable:/A/B:S   \t   \n" );
        BOOST_CHECK_EQUAL( aFullID.getString(), "Variable:/A/B:S" );
        BOOST_CHECK_EQUAL( aFullID.getEntityType(), EntityType::VARIABLE ); 
        BOOST_CHECK( aFullID.getSystemPath() == SystemPath( "/A/B" ) ); 
        BOOST_CHECK_EQUAL( aFullID.getID(), "S" ); 
        BOOST_CHECK( aFullID.isValid() );
    }
    BOOST_CHECK_THROW( FullID( "UNKNOWN:/UNKNOWN:" ), BadFormat );
    { 
        FullID aFullID( "Process::/" );
        BOOST_CHECK_EQUAL( aFullID.getString(), "Process::/" );
        BOOST_CHECK_EQUAL( aFullID.getEntityType(), EntityType::PROCESS ); 
        BOOST_CHECK( aFullID.getSystemPath() == SystemPath( "" ) ); 
        BOOST_CHECK_EQUAL( aFullID.getID(), "/" ); 
        BOOST_CHECK( aFullID.isValid() ); // XXX: is this OK?
    }
}

BOOST_AUTO_TEST_CASE(testFullIDAssignment)
{
    FullID aFullID( "Variable:/test:TEST" );
    FullID anAssignee( EntityType::NONE, "", "" );

    anAssignee = aFullID;
    BOOST_CHECK_EQUAL( anAssignee.getString(), "Variable:/test:TEST" );
    BOOST_CHECK_EQUAL( anAssignee.getEntityType(), EntityType::VARIABLE ); 
    BOOST_CHECK( anAssignee.getSystemPath() == SystemPath( "/test" ) ); 
    BOOST_CHECK_EQUAL( anAssignee.getID(), "TEST" ); 
    BOOST_CHECK( anAssignee.isValid() );
    anAssignee.setEntityType( EntityType::PROCESS );
    anAssignee.setID( "TEST2" );
    BOOST_CHECK_EQUAL( anAssignee.getEntityType(), EntityType::PROCESS );
    BOOST_CHECK_EQUAL( anAssignee.getID(), "TEST2" );
    BOOST_CHECK_EQUAL( aFullID.getEntityType(), EntityType::VARIABLE );
    BOOST_CHECK_EQUAL( aFullID.getID(), "TEST" );
}
/*
void test()
{
     FullID aFullID( "       \t  \n  Variable:/A/B:S   \t   \n" );

     FullID aFullID2( aFullID
     std::cout << aFullID2.getString() << std::endl;

     FullID aFullID3( "Process:/:R"
     std::cout << aFullID3.getString() << std::endl;
     aFullID3 = aFullID2;
     std::cout << aFullID3.getString() << std::endl;

     std::cout << "\n::::::::::" << std::endl;
     //      FullPN aFullPN( 1,aFullID.getSystemPath(),"/", "PNAME"

     FullPN aFullPN( "       \t  \n  Variable:/A/B:S:PNAME   \t   \n"
     std::cout << aFullPN.getString() << std::endl;
     std::cout << aFullPN.getEntityType() << std::endl;
     std::cout << aFullPN.getSystemPath().getString() << std::endl;
     std::cout << aFullPN.getID() << std::endl;
     std::cout << aFullPN.getPropertyName() << std::endl;
     std::cout << aFullPN.isValid() << std::endl;

     FullPN aFullPN2( aFullPN
     std::cout << aFullPN2.getString() << std::endl;

     FullPN aFullPN3( "Process:/:R:P"
     std::cout << aFullPN3.getString() << std::endl;
     aFullPN3 = aFullPN2;
     std::cout << aFullPN3.getString() << std::endl;
}
*/
