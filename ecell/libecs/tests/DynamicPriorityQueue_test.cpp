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

#include "DynamicPriorityQueue.hpp"

#include <gsl/gsl_rng.h>

#include <iostream>

namespace libecs
{

template<typename T>
class DynamicPriorityQueueTest
{
    typedef DynamicPriorityQueue<T> DPQ;
    typedef typename DPQ::Index Index;
    typedef std::vector<Index> IndexVector;

public:
    void testConstruction()
    {
        DPQ dpq;

        BOOST_CHECK( dpq.isEmpty() );
        BOOST_CHECK( dpq.checkConsistency() );
    }

    void testClear()
    {
        DPQ dpq;
        typedef typename DPQ::Index Index;

        dpq.pushItem( 1 );
        dpq.pushItem( 20 );
        dpq.pushItem( 50 );

        BOOST_CHECK_EQUAL( Index( 3 ), dpq.getSize() );

        dpq.clear();

        BOOST_CHECK( dpq.isEmpty() );
        BOOST_CHECK( dpq.checkConsistency() );

        dpq.pushItem( 2 );
        dpq.pushItem( 20 );
        dpq.pushItem( 30 );

        BOOST_CHECK_EQUAL( Index( 3 ), dpq.getSize() );

        dpq.clear();

        BOOST_CHECK( dpq.isEmpty() );
        BOOST_CHECK( dpq.checkConsistency() );
    }

    void testPush()
    {
        DPQ dpq;

        dpq.pushItem( 1 );

        BOOST_CHECK( dpq.checkConsistency() );
        BOOST_CHECK( dpq.getTopItem() == 1.0 );
    }

    void testDuplicatedItems()
    {
        DPQ dpq;

        dpq.pushItem( 1 );
        dpq.pushItem( 2 );
        dpq.pushItem( 1 );
        dpq.pushItem( 2 );

        BOOST_CHECK( dpq.checkConsistency() );

        BOOST_CHECK( dpq.getTopItem() == 1 );
        dpq.popItem();
        BOOST_CHECK( dpq.getTopItem() == 1 );
        dpq.popItem();
        BOOST_CHECK( dpq.getTopItem() == 2 );
        dpq.popItem();
        BOOST_CHECK( dpq.getTopItem() == 2 );
        dpq.popItem();

        BOOST_CHECK( dpq.isEmpty() );

        BOOST_CHECK( dpq.checkConsistency() );
    }

    void testSimpleSorting()
    {
        DPQ dpq;

        const int MAXI( 100 );
        for( int i( MAXI ); i != 0  ; --i )
        {
            dpq.pushItem( i );
        }

        BOOST_CHECK( dpq.checkConsistency() );

        int n( 0 );
        while( ! dpq.isEmpty() )
        {
            ++n;
            BOOST_CHECK_EQUAL( n, dpq.getTopItem() );
            dpq.popItem();
        }

        BOOST_CHECK_EQUAL( MAXI, n );

        BOOST_CHECK( dpq.isEmpty() );
        BOOST_CHECK( dpq.checkConsistency() );
    }

    void testInterleavedSorting()
    {
        DPQ dpq;
        typedef typename DPQ::Index Index;

        const Index MAXI( 101 );
        for( int i( MAXI-1 ); i != 0  ; i-=2 )
        {
            dpq.pushItem( i );
        }

        for( int i( MAXI ); i != -1  ; i-=2 )
        {
            dpq.pushItem( i );
        }

        BOOST_CHECK_EQUAL( MAXI, dpq.getSize() );

        BOOST_CHECK( dpq.checkConsistency() );

        int n( 0 );
        while( ! dpq.isEmpty() )
        {
            ++n;
            BOOST_CHECK_EQUAL( n, dpq.getTopItem() );
            dpq.popItem();
        }

        BOOST_CHECK_EQUAL( MAXI, Index( n ) );

        BOOST_CHECK( dpq.isEmpty() );
        BOOST_CHECK( dpq.checkConsistency() );
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
            BOOST_TEST_SUITE("DPQ testsuites");

    {
        typedef libecs::DynamicPriorityQueueTest<int> IntegerDPQTest;
        boost::unit_test::test_suite* suite =
                BOOST_TEST_SUITE( "DynamicPriorityQueue<int>" );
        boost::shared_ptr<IntegerDPQTest> inst( new IntegerDPQTest() );

        add_test( IntegerDPQTest, testConstruction );
        add_test( IntegerDPQTest, testClear );
        add_test( IntegerDPQTest, testPush );
        add_test( IntegerDPQTest, testDuplicatedItems );
        add_test( IntegerDPQTest, testSimpleSorting );
        add_test( IntegerDPQTest, testInterleavedSorting );

        suites->add(suite);
    }

    {
        typedef libecs::DynamicPriorityQueueTest<double> DoubleDPQTest;
        boost::unit_test::test_suite* suite =
                BOOST_TEST_SUITE( "DynamicPriorityQueue<double>" );
        boost::shared_ptr<DoubleDPQTest> inst(new DoubleDPQTest());

        add_test( DoubleDPQTest, testConstruction );
        add_test( DoubleDPQTest, testClear );
        add_test( DoubleDPQTest, testPush );
        add_test( DoubleDPQTest, testDuplicatedItems );
        add_test( DoubleDPQTest, testSimpleSorting );
        add_test( DoubleDPQTest, testInterleavedSorting );

        suites->add(suite);
    }

    return suites;
}
