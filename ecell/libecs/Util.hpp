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

#ifndef __UTIL_HPP
#define __UTIL_HPP
#include <stdlib.h>
#include <sstream>
#include <functional>
#include <limits>

#include <boost/version.hpp>

#if BOOST_VERSION >= 103200  // for boost-1.32.0 or later.
#    include <boost/numeric/conversion/cast.hpp>
#else                        // use this instead for boost-1.31 or earlier.
#    include <boost/cast.hpp>
#endif


#include <boost/lexical_cast.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

#include "libecs.hpp"
#include "Exceptions.hpp"

namespace libecs
{

  /** @addtogroup util The Utilities.
   Utilities.

   @ingroup libecs
   @{ 
   */ 

  /** @file */

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
    BOOST_STATIC_ASSERT( ( boost::is_same<String,GIVEN>::value ||
			   boost::is_same<String,NEW>::value ) );

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



  /**
     Erase white space characters ( ' ', '\t', and '\n' ) from a string
  */
  void eraseWhiteSpaces( StringRef str );

  template < class T >
  struct PtrGreater
  {
    bool operator()( T x, T y ) const { return *y < *x; }
  };


  template < class T >
  struct PtrLess
  {
    bool operator()( T x, T y ) const { return *y > *x; }
  };



  /**
     Check if aSequence's size() is within [ aMin, aMax ].  

     If not, throw a RangeError exception.

  */

  template <class Sequence>
  void checkSequenceSize( const Sequence& aSequence, 
			  const typename Sequence::size_type aMin, 
			  const typename Sequence::size_type aMax )
  {
    const typename Sequence::size_type aSize( aSequence.size() );
    if( aSize < aMin || aSize > aMax )
      {
	throwSequenceSizeError( aSize, aMin, aMax );
      }
  }


  /**
     Check if aSequence's size() is at least aMin.

     If not, throw a RangeError exception.

  */

  template <class Sequence>
  void checkSequenceSize( const Sequence& aSequence, 
			  const typename Sequence::size_type aMin )
  {
    const typename Sequence::size_type aSize( aSequence.size() );
    if( aSize < aMin )
      {
	throwSequenceSizeError( aSize, aMin );
      }
  }


  ///@internal
  LIBECS_API void throwSequenceSizeError( const size_t aSize, 
			       const size_t aMin, const size_t aMax );

  ///@internal
  LIBECS_API void throwSequenceSizeError( const size_t aSize, const size_t aMin );


  /**
     Form a 'for' loop over a STL sequence.

     Use this like:

     FOR_ALL( std::vector<int>, anIntVector )
     {
       int anInt( *i ); // the iterator is 'i'.
       ...
     }

     @arg SEQCLASS the classname of the STL sequence. 
     @arg SEQ the STL sequence.
  */

#define FOR_ALL( SEQCLASS, SEQ )\
  for( SEQCLASS ::const_iterator i( (SEQ) .begin() ) ;\
      i != (SEQ) .end() ; ++i )



  /**
     For each 'second' member of element in a sequence, call a given method.

     @note This will be deprecated.  Use select2nd instead.

     @arg SEQCLASS the classname of the STL sequence. 
     @arg SEQ the STL sequence.
     @arg METHOD the name of the method.
     
     @see FOR_ALL
  */

#define FOR_ALL_SECOND( SEQCLASS, SEQ, METHOD )\
  FOR_ALL( SEQCLASS, SEQ )\
    { (*i).second-> METHOD (); }




  template< typename T >
  inline const T nullValue()
  {
    return 0;
  }

  template<>
  inline const Real nullValue()
  {
    return 0.0;
  }

  template<>
  inline const String nullValue()
  {
    return String();
  }

  template< class NEW, class GIVEN >
  class StaticCaster
    :
    std::unary_function< GIVEN, NEW >
  {
  public:
    inline NEW operator()( const GIVEN& aValue )
    {
      BOOST_STATIC_ASSERT( ( boost::is_convertible<GIVEN,NEW>::value ) );
      return static_cast<NEW>( aValue );
    }
  };

  template< class NEW, class GIVEN >
  class DynamicCaster
    :
    std::unary_function< GIVEN, NEW >
  {
  public:
    NEW operator()( const GIVEN& aPtr )
    {
      NEW aNew( dynamic_cast<NEW>( aPtr ) );
      if( aNew != NULLPTR )
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
  class LexicalCaster
    :
    std::unary_function< GIVEN, NEW >
  {
  public:
    const NEW operator()( const GIVEN& aValue )
    {
      return stringCast<NEW>( aValue );
    }
  };




  template< class NEW, class GIVEN >
  class NumericCaster
    :
    std::unary_function< GIVEN, NEW >
  {
  public:
    inline NEW operator()( GIVEN aValue )
    {
      return boost::numeric_cast<NEW>( aValue );
    }
  };



  /**
     These functions are prepared for ExpressionFluxProcess
     and are used in it. asinh, acosh and atanh are not available in 
     MS Windows (MinGW).
  */


  template <typename T>
  Real real_not( T n )
  {
    if( n == 0 )
      {
	return 1.0;
      }
    else
      {
	return 0.0;
      }
  }

  template <typename T>
  Real real_eq( T n1, T n2 )
  {
    if( n1 == n2 )
      {
	return 1.0;
      }
    else
      {
	return 0.0;
      }
  }

  template <typename T>
  Real real_neq( T n1, T n2 )
  {
    if( n1 == n2 )
      {
	return 0.0;
      }
    else
      {
	return 1.0;
      }
  }

  template <typename T>
  Real real_gt( T n1, T n2 )
  {
    if( n1 > n2 )
      {
	return 1.0;
      }
    else
      {
	return 0.0;
      }
  }

  template <typename T>
  Real real_lt( T n1, T n2 )
  {
    if( n1 < n2 )
      {
	return 1.0;
      }
    else
      {
	return 0.0;
      }
  }

  template <typename T>
  Real real_geq( T n1, T n2 )
  {
    if( n1 >= n2 )
      {
	return 1.0;
      }
    else
      {
	return 0.0;
      }
  }

  template <typename T>
  Real real_leq( T n1, T n2 )
  {
    if( n1 <= n2 )
      {
	return 1.0;
      }
    else
      {
	return 0.0;
      }
  }

  template <typename T>
  Real real_and( T n1, T n2 )
  {
    if( ( n1 != 0 ) && ( n2 != 0 ) )
      {
	return 1.0;
      }
    else
      {
	return 0.0;
      }
  }

  template <typename T>
  Real real_or( T n1, T n2 )
  {
    if( ( n1 != 0 ) || ( n2 != 0 ) )
      {
	return 1.0;
      }
    else
      {
	return 0.0;
      }
  }

  template <typename T>
  Real real_xor( T n1, T n2 )
  {
    if( ( n1 != 0 ) && !( n2 != 0 ) )
      {
	return 1.0;
      }
    else
      {
	return 0.0;
      }
  }

  template <typename T>
  T asinh( T n )
  {
    return log( n + sqrt( n * n + 1 ) );
  }

  template <typename T>
  T acosh( T n )
  {
    return log( n - sqrt( n * n - 1 ) );
  }

  template <typename T>
  T atanh( T n )
  {
    return 0.5 * log( ( 1 + n ) / ( 1 - n ) );
  }

  template <typename T>
  T sec( T n )
  {
    return 1 / cos( n );
  }

  template <typename T>
  T csc( T n )
  {
    return 1 / sin( n );
  }

  template <typename T>
  T cot( T n )
  {
    return 1 / tan( n );
  }

  template <typename T>
  T asec( T n )
  {
    return 1 / acos( n );
  }

  template <typename T>
  T acsc( T n )
  {
    return 1 / asin( n );
  }

  template <typename T>
  T acot( T n )
  {
    return 1 / atan( n );
  }

  template <typename T>
  T sech( T n )
  {
    return 1 / cosh( n );
  }
  
  template <typename T>
  T csch( T n )
  {
    return 1 / sinh( n );
  }
  
  template <typename T>
  T coth( T n )
  {
    return 1 / tanh( n );
  }
  
  template <typename T>
  T asech( T n )
  {
    return 1 / acosh( n );
  }

  template <typename T>
  T acsch( T n )
  {
    return 1 / asinh( n );
  }

  template <typename T>
  T acoth( T n )
  {
    return 1 / atanh( n );
  }

  template <typename T>
  T fact( T n )
  {
    if( n <= 1 )
      return 1;
    else
      return n * fact( n-1 );
  }
  
  const Polymorph convertStringMapToPolymorph( StringMap const& aMap );


  //@}

} // namespace libecs


#endif /* __UTIL_HPP */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/

