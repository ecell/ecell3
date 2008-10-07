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

#define BOOST_TEST_MODULE "SystemPath"

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <boost/test/test_case_template.hpp>
#include <boost/preprocessor/stringize.hpp>

#include "SystemPath.hpp"
#include "Exceptions.hpp"

#include <iostream>

using libecs::SystemPath;
using libecs::BadFormat;

BOOST_AUTO_TEST_CASE(testConstruction)
{
    {
        SystemPath aSystemPath( SystemPath::parse( "" ) );
        BOOST_CHECK_EQUAL( aSystemPath.asString(), "" );
    }
    {
        SystemPath aSystemPath( SystemPath::parse( "/" ) );
        BOOST_CHECK_EQUAL( aSystemPath.asString(), "/" );
    }
    {
        SystemPath aSystemPath( SystemPath::parse( "/A/B/C/" ) );
        BOOST_CHECK_EQUAL( aSystemPath.asString(), "/A/B/C" );
    }
    {
        SystemPath aSystemPath( SystemPath::parse( "." ) );
        BOOST_CHECK_EQUAL( aSystemPath.asString(), "." );
    }
    {
        SystemPath aSystemPath( SystemPath::parse( ".." ) );
        BOOST_CHECK_EQUAL( aSystemPath.asString(), ".." );
    }
    {
        SystemPath aSystemPath(
            SystemPath::parse( "   \t  /A/BB/CCC/../DDDD/EEEEEE    \t \n  " ) );
        BOOST_CHECK_EQUAL( aSystemPath.asString(),
                           "/A/BB/CCC/../DDDD/EEEEEE" );
    }
}

BOOST_AUTO_TEST_CASE(testCopyConstruction)
{
    SystemPath aSystemPath(
        SystemPath::parse( "   \t  /A/BB/CCC/../DDDD/EEEEEE    \t \n  " ) );
    SystemPath aSystemPath2( aSystemPath );
    BOOST_CHECK_EQUAL( aSystemPath2.asString(),
                       "/A/BB/CCC/../DDDD/EEEEEE" );
}

BOOST_AUTO_TEST_CASE(testPop)
{
    {
        SystemPath aSystemPath( SystemPath::parse( "/A/BB/CCC/../DDDD/EEEEEE" ) );
        aSystemPath.pop();
        BOOST_CHECK_EQUAL( aSystemPath.asString(),
                           "/A/BB/CCC/../DDDD" );
        aSystemPath.pop();
        BOOST_CHECK_EQUAL( aSystemPath.asString(),
                           "/A/BB/CCC/.." );
        aSystemPath.pop();
        BOOST_CHECK_EQUAL( aSystemPath.asString(),
                           "/A/BB/CCC" );
        aSystemPath.pop();
        BOOST_CHECK_EQUAL( aSystemPath.asString(),
                           "/A/BB" );
        aSystemPath.pop();
        BOOST_CHECK_EQUAL( aSystemPath.asString(),
                           "/A" );
    }

    {
        SystemPath aSystemPath( SystemPath::parse( "/" ) );
        aSystemPath.pop();
        BOOST_CHECK_EQUAL( aSystemPath.asString(), "" );
    }
}

BOOST_AUTO_TEST_CASE(testIsAbsolute)
{
    BOOST_CHECK( SystemPath::parse( "/A/BB/CCC/../DDDD/EEEEEE" ).isAbsolute() );
    BOOST_CHECK( !SystemPath::parse( "A/.." ).isAbsolute() );
    BOOST_CHECK( !SystemPath::parse( "././." ).isAbsolute() );
    BOOST_CHECK( SystemPath::parse( "/" ).isAbsolute() );
    BOOST_CHECK( SystemPath::parse( "" ).isAbsolute() );
    BOOST_CHECK( SystemPath::parse( "/." ).isAbsolute() );
    BOOST_CHECK( !SystemPath::parse( ".." ).isAbsolute() );
    BOOST_CHECK( SystemPath::parse( "/.." ).isAbsolute() );
}

BOOST_AUTO_TEST_CASE(testIsRoot)
{
    BOOST_CHECK( !SystemPath::parse( "/A/BB/CCC/../DDDD/EEEEEE" ).isRoot() );
    BOOST_CHECK( !SystemPath::parse( "A/.." ).isRoot() );
    BOOST_CHECK( SystemPath::parse( "/" ).isRoot() );
    BOOST_CHECK( !SystemPath::parse( "" ).isRoot() );
    BOOST_CHECK( !SystemPath::parse( "A" ).isRoot() );
    BOOST_CHECK( !SystemPath::parse( "A/B" ).isRoot() );
    BOOST_CHECK( SystemPath::parse( "/./././" ).isRoot() );
    BOOST_CHECK( SystemPath::parse( "/A/B/../.." ).isRoot() );
    BOOST_CHECK( !SystemPath::parse( ".." ).isRoot() );
    BOOST_CHECK( SystemPath::parse( "/.." ).isRoot() );
}
