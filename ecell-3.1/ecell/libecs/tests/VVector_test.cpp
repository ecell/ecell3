//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2010 Keio University
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
// written by Moriyoshi Koizumi
//

#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif

#define BOOST_TEST_MODULE "VVector"
#define BOOST_TEST_ALTERNATIVE_INIT_API 1
#define BOOST_TEST_NO_MAIN

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <boost/test/test_case_template.hpp>
#include <boost/preprocessor/stringize.hpp>

#include "VVector.h"

#include <iostream>

namespace libecs
{

template<typename T>
class VVectorTest
{
    typedef vvector<T> Vector;

    Vector* create()
    {
        return new Vector();
    }

public:
    void test()
    {
        Vector* v = create();

        for (int i = 0; i < 1024; i++)
        {
            v->push_back( i );
        }

        delete v;
    }
};

} // namespace libecs

bool my_init_unit_test()
{
#   define add_test(klass, method) \
        suite->add(boost::unit_test::make_test_case<klass>( \
            &klass::method, \
            BOOST_PP_STRINGIZE(klass) "::" BOOST_PP_STRINGIZE(method), \
            inst))

    {
        typedef libecs::VVectorTest<int> IntVVectorTest;
        boost::unit_test::test_suite* suite =
                BOOST_TEST_SUITE( "VVectorTest<int>" );
        boost::shared_ptr<IntVVectorTest> inst( new IntVVectorTest() );

        add_test( IntVVectorTest, test );
        boost::unit_test::framework::master_test_suite().add(suite);
    }

    return true;
}

int main( int argc, char **argv )
{
    return ::boost::unit_test::unit_test_main( &my_init_unit_test, argc, argv );
}
