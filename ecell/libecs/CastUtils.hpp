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

#ifndef __CASTUTILS_HPP
#define __CASTUTILS_HPP

#include <functional>
#include <boost/version.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>

#if BOOST_VERSION >= 103200  // for boost-1.32.0 or later.
#    include <boost/numeric/conversion/cast.hpp>
#else                        // use this instead for boost-1.31 or earlier.
#    include <boost/cast.hpp>
#endif

#include "Exceptions.hpp"

/**
   @addtogroup util The Utilities.
   Utilities.

   @ingroup libecs
 */
/** @file */
/** @{ */

namespace libecs {

/**
    A universal to String / from String converter.

    Two usages:
    - stringCast( VALUE )        -- convert VALUE to a string.
    - stringCast<TYPE>( STRING ) -- convert STRING to a TYPE object.

    This is a thin wrapper over boost::lexical_cast.
    This stringCast template function has some specializations for
    common numeric types such as Real and Integer are defined, and
    use of this instead of boost::lexical_cast for those types
    can reduce resulting binary size.
*/
template< typename NEW, typename GIVEN >
const NEW stringCast( const GIVEN& aValue )
{
    BOOST_STATIC_ASSERT( ( boost::is_same<String, GIVEN>::value ||
                           boost::is_same<String, NEW>::value ) );

    return boost::lexical_cast<NEW>( aValue );
}

///@internal
template< typename GIVEN >
const String stringCast( const GIVEN& aValue )
{
    return stringCast<String,GIVEN>( aValue );
}

#define __STRINGCAST_SPECIALIZATION_DECL( NEW, GIVEN )\
  template<> LIBECS_API const NEW stringCast<NEW,GIVEN>( const GIVEN& )

__STRINGCAST_SPECIALIZATION_DECL( String, Real );
__STRINGCAST_SPECIALIZATION_DECL( String, HighReal );
__STRINGCAST_SPECIALIZATION_DECL( String, Integer );
__STRINGCAST_SPECIALIZATION_DECL( String, UnsignedInteger );
__STRINGCAST_SPECIALIZATION_DECL( Real, String );
__STRINGCAST_SPECIALIZATION_DECL( HighReal, String );
__STRINGCAST_SPECIALIZATION_DECL( Integer, String );
__STRINGCAST_SPECIALIZATION_DECL( UnsignedInteger, String );
// __STRINGCAST_SPECIALIZATION_DECL( String, String );

#undef __STRINGCAST_SPECIALIZATION_DECL

template< class NEW, class GIVEN >
struct StaticCaster
       : std::unary_function< GIVEN, NEW >
{
    typedef GIVEN argument_type;
    typedef NEW result_type;

    inline NEW operator()( const GIVEN& aValue )
    {
        BOOST_STATIC_ASSERT( ( boost::is_convertible<GIVEN,NEW>::value ) );
        return static_cast<NEW>( aValue );
    }
};

template< class NEW, class GIVEN >
struct DynamicCaster
        : std::unary_function< GIVEN, NEW >
{
    typedef GIVEN argument_type;
    typedef NEW result_type;

    NEW operator()( const GIVEN& aPtr )
    {
        NEW aNew( dynamic_cast<NEW>( aPtr ) );
        if ( aNew != NULLPTR )
        {
            return aNew;
        }
        else
        {
            THROW_EXCEPTION( TypeError, "dynamic cast failed." );
        }
    }
};

template< class NEW, class GIVEN >
struct ReinterpretCaster
        : std::unary_function< GIVEN, NEW >
{
    typedef GIVEN argument_type;
    typedef NEW result_type;

    NEW operator()( const GIVEN& aPtr )
    {
        return reinterpret_cast<NEW>( aPtr );
    }
};

template< class NEW, class GIVEN >
struct LexicalCaster
        : std::unary_function< GIVEN, NEW >
{
    typedef GIVEN argument_type;
    typedef NEW result_type;

    const NEW operator()( const GIVEN& aValue )
    {
        return stringCast<NEW>( aValue );
    }
};

template< class NEW, class GIVEN >
struct NumericCaster: std::unary_function< GIVEN, NEW >
{
    typedef GIVEN argument_type;
    typedef NEW result_type;

    inline NEW operator()( GIVEN aValue )
    {
        return boost::numeric_cast<NEW>( aValue );
    }
};

} // namespace libecs

/** @} */

#endif /* __CASTUTILS_HPP */
