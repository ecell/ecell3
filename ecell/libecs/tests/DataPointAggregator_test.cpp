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

#include "DataPoint.hpp"
#include <iostream>

namespace libecs
{

template<typename CharT, typename Traits>
inline std::basic_ostream<CharT, Traits>&
operator <<(std::basic_ostream<CharT, Traits>& strm, DataPoint const& dp)
{
    std::string buf;
    strm << dp.getTime() << ":" << dp.getValue();
    return strm;
}

class DataPointAggregatorTest
{
public:
    void testAggregate()
    {
        DataPointAggregator dpa;
        dpa.aggregate( DataPoint(  3, 4 ) );
        BOOST_CHECK_EQUAL( dpa.getLastPoint(), DataPoint(  3, 4 ) );
        BOOST_CHECK_EQUAL( dpa.getData().getMin(), 4 );
        BOOST_CHECK_EQUAL( dpa.getData().getMax(), 4 );
        BOOST_CHECK_EQUAL( dpa.getData().getAvg(), 4 );
        dpa.aggregate( DataPoint(  5, 6 ) );
        BOOST_CHECK_EQUAL( dpa.getLastPoint(), DataPoint(  5, 6 ) );
        BOOST_CHECK_EQUAL( dpa.getData().getMin(), 4 );
        BOOST_CHECK_EQUAL( dpa.getData().getMax(), 6 );
        BOOST_CHECK_EQUAL( dpa.getData().getAvg(), 4 );
        dpa.aggregate( DataPoint(  7, 8 ) );
        BOOST_CHECK_EQUAL( dpa.getLastPoint(), DataPoint(  7, 8 ) );
        BOOST_CHECK_EQUAL( dpa.getData().getMin(), 4 );
        BOOST_CHECK_EQUAL( dpa.getData().getMax(), 8 );
        BOOST_CHECK_EQUAL( dpa.getData().getAvg(), 5 );
        dpa.aggregate( DataPoint(  9, 6 ) );
        BOOST_CHECK_EQUAL( dpa.getLastPoint(), DataPoint(  9, 6 ) );
        BOOST_CHECK_EQUAL( dpa.getData().getMin(), 4 );
        BOOST_CHECK_EQUAL( dpa.getData().getMax(), 8 );
        BOOST_CHECK_EQUAL( dpa.getData().getAvg(), 6 );
        dpa.aggregate( DataPoint( 11, 4 ) );
        BOOST_CHECK_EQUAL( dpa.getData().getMin(), 4 );
        BOOST_CHECK_EQUAL( dpa.getData().getMax(), 8 );
        BOOST_CHECK_EQUAL( dpa.getData().getAvg(), 6 );
        dpa.aggregate( DataPoint( 13, 6 ) );
        BOOST_CHECK_EQUAL( dpa.getData().getMin(), 4 );
        BOOST_CHECK_EQUAL( dpa.getData().getMax(), 8 );
        BOOST_CHECK_EQUAL( dpa.getData().getAvg(), 5.6 );
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
            BOOST_TEST_SUITE("DataPointAggregator testsuites");

    {
        boost::unit_test::test_suite* suite =
                BOOST_TEST_SUITE( "DataPointAggregator" );
        boost::shared_ptr<libecs::DataPointAggregatorTest>
            inst( new libecs::DataPointAggregatorTest() );

        add_test( libecs::DataPointAggregatorTest, testAggregate );

        suites->add(suite);
    }

    return suites;
}

