//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2008 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
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
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#ifndef __LIBECS_HPP
#define __LIBECS_HPP

#include "Defs.hpp"

#include <list>
#include <vector>
#include <map>

#include <boost/smart_ptr.hpp>
#include <boost/call_traits.hpp>

namespace libecs
{

// Types
template <typename T>
class Param
{
public:
    typedef typename boost::call_traits<T>::param_type type;
};

// String
DECLARE_TYPE( ::std::string, String );

// Numeric types
DECLARE_TYPE( long int, Integer );
DECLARE_TYPE( unsigned long int, UnsignedInteger );
DECLARE_TYPE( double, Real );
DECLARE_TYPE( HIGHREAL_TYPE, HighReal );
DECLARE_TYPE( Real, Time );
DECLARE_TYPE( Real, TimeDifference );

typedef Param<Integer>::type IntegerParam;
typedef Param<UnsignedInteger>::type UnsignedIntegerParam;
typedef Param<Real>::type RealParam;
typedef Param<HighReal>::type HighRealParam;
typedef Param<Time>::type TimeParam;
typedef Param<TimeDifference>::type TimeDifferenceParam;

const Real N_A( 6.0221367e+23 );
const Real N_A_R( 1.0 / N_A );

#if defined( FP_FAST_FMA )
inline const Real FMA( const Real a, const Real b, const Real c )
{
    return ::fma( a, b, c );
}
#else
inline const Real FMA( const Real a, const Real b, const Real c )
{
    return a * b + c;
}
#endif /* defined( FP_FAST_FMA ) */

/**
 Converts each type into a unique, insipid type.
 Invocation Type2Type<T> where T is a type.
 Defines the type OriginalType which maps back to T.

 taken from loki library.

 @ingroup util
*/

template <typename T>
struct Type2Type
{
    typedef T OriginalType;
};

template<typename T>
inline T safe_add( const T& o1, const T& o2 )
{
    T r = o1 + o2;
    BOOST_ASSERT( r >= o1 || r >= o2 );
    return r;
}

template<typename T1, typename T2>
inline T1 safe_mul( const T1& o1, const T2& o2 )
{
    T1 r = o1 * o2;
    BOOST_ASSERT( sizeof( o1 ) >= sizeof( double ) || static_cast<T1>( static_cast<double>( o1 ) * o2 ) == r );
    return r;
}

template<typename Td, typename Ts>
inline Td safe_int_cast( const Ts& v )
{
    const Td retval = static_cast<Td>( v );
    BOOST_ASSERT( v == static_cast<Ts>( retval ) );
    return retval;
}

/** @defgroup libecs The Libecs library
 * The libecs library
 * @{
 */

/** @file */


LIBECS_API extern int const MAJOR_VERSION;
LIBECS_API extern int const MINOR_VERSION;
LIBECS_API extern int const MICRO_VERSION;

LIBECS_API extern char const* const VERSION_STRING;


LIBECS_API const int getMajorVersion();
LIBECS_API const int getMinorVersion();
LIBECS_API const int getMicroVersion();
LIBECS_API const std::string getVersion();

LIBECS_API bool initialize();
LIBECS_API void finalize();
LIBECS_API void setDMSearchPath( const String& path );
LIBECS_API const String getDMSearchPath();

/** @} */

} // namespace libecs

#endif // __LIBECS_HPP


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
