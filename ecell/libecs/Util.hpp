//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-CELL is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-CELL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-CELL -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Kouichi Takahashi <shafi@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//

#ifndef ___UTIL_H___
#define ___UTIL_H___
#include <stdlib.h>
#include <sstream>
#include <functional>
#include <limits>

#include "Defs.hpp"
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
      Universal String -> object converter.
      Real and Int specializations are defined in Util.cpp.
      Conversion to the other classes are conducted using 
      istrstream.
  */

  // FIXME: should be a static function object? to reduce initialization cost

  template <typename T> 
  const T stringTo( StringCref str )
  {
    std::istringstream ist( str.c_str() );
    T aT;
    ist >> aT;
    return aT;
  }

  /// A specialization of stringTo for Real
  template<> const Real stringTo<Real>( StringCref str );

  /// A specialization of stringTo for Int
  template<> const Int   stringTo<Int>( StringCref str );

  /// A specialization of stringTo for UnsignedInt
  template<> const UnsignedInt  stringTo<UnsignedInt>( StringCref str );

  /**
     Any to String converter function template.
     Using ostringstream by default. A specialization for Real type
     is also defined.
  */

  // FIXME: should be a static function object? to reduce initialization cost

  template <typename T> inline const String toString( const T& t )
  {
    std::ostringstream os;
    os << t;
    os << std::ends;
    return os.str();
  }

  /// A specialization of toString for Real
  template<> const String toString<Real>( RealCref f );


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
  void throwSequenceSizeError( const Int aSize, 
			       const Int aMin, const Int aMax );

  ///@internal
  void throwSequenceSizeError( const Int aSize, const Int aMin );


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
  class DynamicCaster
    :
    std::unary_function< GIVEN, NEW >
  {
  public:
    NEW operator()( GIVEN aPtr )
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


  //@}

} // namespace libecs


#endif /* ___UTIL_H___ */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/

