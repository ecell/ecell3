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

        dpq.push( 1 );
        dpq.push( 20 );
        dpq.push( 50 );

        BOOST_CHECK_EQUAL( Index( 3 ), dpq.getSize() );

        dpq.clear();

        BOOST_CHECK( dpq.isEmpty() );
        BOOST_CHECK( dpq.checkConsistency() );

        dpq.push( 2 );
        dpq.push( 20 );
        dpq.push( 30 );

        BOOST_CHECK_EQUAL( Index( 3 ), dpq.getSize() );

        dpq.clear();

        BOOST_CHECK( dpq.isEmpty() );
        BOOST_CHECK( dpq.checkConsistency() );
    }

    void testPush()
    {
        DPQ dpq;

        dpq.push( 1 );

        BOOST_CHECK( dpq.checkConsistency() );
        BOOST_CHECK( dpq.getTop() == 1.0 );
    }

    void testDuplicatedItems()
    {
        DPQ dpq;

        dpq.push( 1 );
        dpq.push( 2 );
        dpq.push( 1 );
        dpq.push( 2 );

        BOOST_CHECK( dpq.checkConsistency() );

        BOOST_CHECK( dpq.getTop() == 1 );
        dpq.popTop();
        BOOST_CHECK( dpq.getTop() == 1 );
        dpq.popTop();
        BOOST_CHECK( dpq.getTop() == 2 );
        dpq.popTop();
        BOOST_CHECK( dpq.getTop() == 2 );
        dpq.popTop();

        BOOST_CHECK( dpq.isEmpty() );

        BOOST_CHECK( dpq.checkConsistency() );
    }

    void testSimpleSorting()
    {
        DPQ dpq;

        const int MAXI( 100 );
        for( int i( MAXI ); i != 0  ; --i )
        {
            dpq.push( i );
        }

        BOOST_CHECK( dpq.checkConsistency() );

        int n( 0 );
        while( ! dpq.isEmpty() )
        {
            ++n;
            BOOST_CHECK_EQUAL( n, dpq.getTop() );
            dpq.popTop();
        }

        BOOST_CHECK_EQUAL( MAXI, n );

        BOOST_CHECK( dpq.isEmpty() );
        BOOST_CHECK( dpq.checkConsistency() );
    }

    void testInterleavedSorting()
    {
        DPQ dpq;

        const Index MAXI( 101 );
        for( int i( MAXI-1 ); i != 0  ; i-=2 )
        {
            dpq.push( i );
        }

        for( int i( MAXI ); i != -1  ; i-=2 )
        {
            dpq.push( i );
        }

        BOOST_CHECK_EQUAL( MAXI, dpq.getSize() );

        BOOST_CHECK( dpq.checkConsistency() );

        int n( 0 );
        while( ! dpq.isEmpty() )
        {
            ++n;
            BOOST_CHECK_EQUAL( n, dpq.getTop() );
            dpq.popTop();
        }

        BOOST_CHECK_EQUAL( MAXI, Index( n ) );

        BOOST_CHECK( dpq.isEmpty() );
        BOOST_CHECK( dpq.checkConsistency() );
    }

    void testReplaceTop()
    {
        DPQ dpq;

        dpq.push( 4 );
        dpq.push( 2 );
        dpq.push( 1 );

        BOOST_CHECK_EQUAL( 1, dpq.getTop() );

        dpq.replaceTop( 3 );

        BOOST_CHECK( dpq.checkConsistency() );
        BOOST_CHECK_EQUAL( 2, dpq.getTop() );

        dpq.popTop();
        BOOST_CHECK_EQUAL( 3, dpq.getTop() );
        dpq.popTop();
        BOOST_CHECK_EQUAL( 4, dpq.getTop() );
        dpq.popTop();


        BOOST_CHECK( dpq.isEmpty() );
        BOOST_CHECK( dpq.checkConsistency() );
    }

    void testReplace()
    {
        DPQ dpq;

        dpq.push( 5 );
        const Index id( dpq.push( 4 ) );
        dpq.push( 3 );
        dpq.push( 1 );

        BOOST_CHECK_EQUAL( 1, dpq.getTop() );

        dpq.replace( id, 2 );  // 4->2

        BOOST_CHECK( dpq.checkConsistency() );
        BOOST_CHECK_EQUAL( 1, dpq.getTop() );

        dpq.popTop();
        BOOST_CHECK_EQUAL( 2, dpq.getTop() );
        dpq.popTop();
        BOOST_CHECK_EQUAL( 3, dpq.getTop() );
        dpq.popTop();
        BOOST_CHECK_EQUAL( 5, dpq.getTop() );
        dpq.popTop();

        BOOST_CHECK( dpq.isEmpty() );
        BOOST_CHECK( dpq.checkConsistency() );
    }

    void testSimpleSortingWithPops()
    {
        DPQ dpq;

        IndexVector idVector;

        const Index MAXI( 100 );
        for( int n( MAXI ); n != 0  ; --n )
        {
            Index id( dpq.push( n ) );
            if( n == 11 || n == 45 )
            {
                idVector.push_back( id );
            }
        }

        BOOST_CHECK( dpq.checkConsistency() );

        BOOST_CHECK_EQUAL( MAXI, dpq.getSize() );

        for( typename IndexVector::const_iterator i( idVector.begin() );
             i != idVector.end(); ++i )
        {
            dpq.pop( *i );
        }

        BOOST_CHECK_EQUAL( MAXI - 2, dpq.getSize() );

        int n( 0 );
        while( ! dpq.isEmpty() )
        {
            ++n;
            if( n == 11 || n == 45 )
            {
                continue; // skip
            }
            BOOST_CHECK_EQUAL( int( n ), dpq.getTop() );
            dpq.popTop();
        }

        BOOST_CHECK_EQUAL( MAXI, Index( n ) );

        BOOST_CHECK( dpq.isEmpty() );
        BOOST_CHECK( dpq.checkConsistency() );
    }

    void testInterleavedSortingWithPops()
    {
        DPQ dpq;

        IndexVector idVector;

        const Index MAXI( 101 );
        for( int n( MAXI-1 ); n != 0  ; n-=2 )
        {
            const Index id( dpq.push( n ) );

            if( n == 12 || n == 46 )
            {
                idVector.push_back( id );
            }
        }

        dpq.pop( idVector.back() );
        idVector.pop_back();

        BOOST_CHECK_EQUAL( MAXI/2 -1, dpq.getSize() );

        BOOST_CHECK( dpq.checkConsistency() );

        for( int n( MAXI ); n != -1  ; n-=2 )
        {
            const Index id( dpq.push( n ) );

            if( n == 17 || n == 81 )
            {
                idVector.push_back( id );
            }
        }

        for( typename IndexVector::const_iterator i( idVector.begin() );
             i != idVector.end(); ++i )
        {
            dpq.pop( *i );
        }

        BOOST_CHECK( dpq.checkConsistency() );
        BOOST_CHECK_EQUAL( MAXI-4, dpq.getSize() );

        int n( 0 );
        while( ! dpq.isEmpty() )
        {
            ++n;
            if( n == 12 || n == 46 || n == 17 || n == 81 )
            {
                continue;
            }
            BOOST_CHECK_EQUAL( n, dpq.getTop() );
            dpq.popTop();
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
        add_test( IntegerDPQTest, testReplace );
        add_test( IntegerDPQTest, testReplaceTop );
        add_test( IntegerDPQTest, testSimpleSortingWithPops );
        add_test( IntegerDPQTest, testInterleavedSortingWithPops );

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
        add_test( DoubleDPQTest, testReplace );
        add_test( DoubleDPQTest, testReplaceTop );
        add_test( DoubleDPQTest, testSimpleSortingWithPops );
        add_test( DoubleDPQTest, testInterleavedSortingWithPops );

        suites->add(suite);
    }

    return suites;
}
