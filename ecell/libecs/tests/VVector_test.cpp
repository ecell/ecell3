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
// written by Moriyoshi Koizumi
//

#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif

#define BOOST_TEST_MODULE "VVector"

#include <cmath>
#include <algorithm>
#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <boost/test/test_case_template.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/mpl/list.hpp>

#include "VVector.hpp"

typedef boost::mpl::list7<char, short, int, long, float, double, long double> scalar_types;

using libecs::VVector;
using libecs::VVectorMaker;

BOOST_AUTO_TEST_CASE_TEMPLATE(testPushBack, T_, scalar_types)
{
    VVector< T_ > *a = VVectorMaker::getInstance().create<T_>();
    typedef typename VVector< T_ >::size_type size_type;
    size_type lim = static_cast< size_type >( std::min(
        std::pow( 2., (int)sizeof( T_ ) * 8 ) - 1,
        65536. ) );

    for ( size_type i = 0; i < lim; ++i )
    {
        a->push_back( i );
    }

    BOOST_CHECK_EQUAL( lim, a->size() );

    for ( size_type i = 0; i < lim; ++i )
    {
        a->push_back( i );
    }

    BOOST_CHECK_EQUAL( lim * 2, a->size() );

    delete a;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(testPushBackAndRef, T_, scalar_types)
{
    VVector< T_ > *a = VVectorMaker::getInstance().create<T_>();
    typedef typename VVector< T_ >::size_type size_type;
    size_type lim = static_cast< size_type >( std::min(
        std::pow( 2., (int)sizeof( T_ ) * 8 ) - 1,
        65536. ) );

    for ( size_type i = 0; i < lim; ++i )
    {
        a->push_back( i );
    }

    BOOST_CHECK_EQUAL( lim, a->size() );

    for ( size_type i = 0; i < lim; ++i )
    {
        BOOST_CHECK_EQUAL( static_cast< T_ >( i ), ( *a )[ i ] );
    }

    for ( size_type i = 0; i < lim; ++i )
    {
        a->push_back( i );
    }

    BOOST_CHECK_EQUAL( lim * 2, a->size() );

    for ( size_type i = 0; i < lim * 2; ++i )
    {
        BOOST_CHECK_EQUAL( static_cast< T_ >( i % lim ), ( *a )[ i ] );
    }

    delete a;
}
