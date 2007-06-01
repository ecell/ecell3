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
// written by Moriyoshi Koizumi
//
#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <boost/test/test_case_template.hpp>
#include <boost/preprocessor/stringize.hpp>
#include "VVector.h"

#include <iostream>
#include <cstdlib>
#include <vector>

namespace libecs
{

template<typename T>
class VVectorTest
{
    typedef VVector<T> Vector;

    char* getTempFileName()
    {
        static const char placeholder[] = "/XXXXXXXX";
        static const size_t placeholder_len = sizeof( placeholder ) - 1;

        const char* tmpdir = libecs_get_temp_dir();
        size_t tmpdir_len = strlen( tmpdir );
        char *tmpfile = new char[ tmpdir_len + placeholder_len + 1 ];
        memcpy( tmpfile, tmpdir, tmpdir_len );
        memcpy( tmpfile + tmpdir_len, placeholder, placeholder_len );
        tmpfile[ tmpdir_len + placeholder_len ] = '\0';

        return tmpfile;
    }

    Vector* create()
    {
        fildes_t fd;
        char* tmpfile = getTempFileName();

        fd = mkstemp( tmpfile );
        if (fd < 0)
        {
            throw IOException( "VVectorTest<>::create()",
                               "Cannot create temporary file");
        }

        try
        {
            return VVectorMaker::getInstance().create<T>( fd, tmpfile );
        }
        catch ( IOException e )
        {
            close( fd );
            unlink( tmpfile );
            delete[] tmpfile;
            throw e;
        }

        // NEVER GET HERE
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

    void testRandomAccess()
    {
        static const int SPACE = 1024 * 1024;
        static const int NUM_SAMPLES = 1024;

        Vector* v = create();
        std::vector<T> realv;

        realv.resize( SPACE );

        for( int i = 0; i < NUM_SAMPLES; i++ )
        {
            int idx = ::rand() % SPACE;

            v->at( idx ) = i;
            realv.at( idx ) = i;
        }

        for( int idx = 0; idx < SPACE; idx++ )
        {
            BOOST_CHECK_EQUAL( v->at( idx ), realv.at( idx ) );
        }

        delete v;
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
            BOOST_TEST_SUITE( "VVector testsuites" );

    {
        typedef libecs::VVectorTest<int> IntVVectorTest;
        boost::unit_test::test_suite* suite =
                BOOST_TEST_SUITE( "VVectorTest<int>" );
        boost::shared_ptr<IntVVector> inst( new IntVVectorTest() );

        add_test( IntVVectorTest, test );
        add_test( IntVVectorTest, testRandomAccess );
        suites->add(suite);
    }

    return suites;
}
