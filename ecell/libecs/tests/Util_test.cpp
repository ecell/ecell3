//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 1996-2007 Keio University
//                Copyright (C) 2005 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Koichi Takahashi
// modified by Moriyoshi Koizumi
//

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <boost/test/test_case_template.hpp>
#include <boost/preprocessor/stringize.hpp>

#include "Util.hpp"
#include "Exceptions.hpp"

#include <iostream>

namespace libecs
{

class UtilTest
{
public:
    void test()
    {
        std::string str( "  \t  a bcde f\tghi\n\t jkl\n \tmnopq     \n   \t " );
        eraseWhiteSpaces( str );
        BOOST_CHECK_EQUAL(str, "abcdefghijklmnopq");
    }
};

} // namespace libecs

boost::unit_test::test_suite* init_unit_test_suite(int argc, char* argv[])
{
#   define add_test(klass, method) \
        suite->add(boost::unit_test::make_test_case<klass>( \
            &klass::method, \
            BOOST_PP_STRINGIZE(klass) "::" BOOST_PP_STRINGIZE(method), \
            inst))
    boost::unit_test::test_suite* suites =
            BOOST_TEST_SUITE( "Util testsuites" );

    {
        boost::unit_test::test_suite* suite =
                BOOST_TEST_SUITE( "Util" );
        boost::shared_ptr<libecs::UtilTest>
            inst( new libecs::UtilTest() );

        add_test( libecs::UtilTest, test );
        suites->add(suite);
    }


    return suites;
}

