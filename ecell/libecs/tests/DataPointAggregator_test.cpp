//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2012 Keio University
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

#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#define BOOST_TEST_MODULE DataPointAggregator

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <boost/test/test_case_template.hpp>
#include <boost/preprocessor/stringize.hpp>

#include "DataPoint.hpp"
#include <iostream>

namespace libecs  {

template<typename CharT, typename Traits>
inline std::basic_ostream<CharT, Traits>&
operator <<(std::basic_ostream<CharT, Traits>& strm, DataPoint const& dp)
{
    std::string buf;
    strm << dp.getTime() << ":" << dp.getValue();
    return strm;
}

BOOST_AUTO_TEST_CASE(testAggregate)
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

} // namespace libecs
