//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2000 Keio University
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

#include "Defs.hpp"
#include "korandom/korandom.h"

namespace libecs
{

  typedef ::korandom_d_c  RandomNumberGenerator;

  /**
     Random number generator
  */
  //FIXME: thread safe?
  extern RandomNumberGenerator* theRandomNumberGenerator; 


  /** 
      Universal String -> object converter.
      Real and Int specializations are defined in Util.cpp.
      Conversion to the other classes are conducted using 
      istrstream.
  */

  // FIXME: should be a static function object? to reduce initialization cost

  template <class T> 
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
     with precision( FLOAT_DIG ) is also defined.
  */

  // FIXME: should be a static function object? to reduce initialization cost

  template <class T> const String toString( const T& t )
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

  /**
     reversed order compare
  */

  template <class T>
  class ReverseCmp
  {
  public:
    bool operator()( const T x, const T y ) const { return x > y; }
  };



  /**
     For each element of a sequence, call a given method.

    There are several different implementations of this.  Actually
    using macro is ugly.  But in core portions of libecs,
    performance is very critical.  

    I've examined assembler codes made by gcc 2.95 and found that
    raw for loops result in the best codes.  The following function
    objects + for_each approach was the second best and almost
    equivalent to the raw loop except that for_each is a template
    function and didn't be inlined by the compiler.

    template <class T>
    class DifferentiateCaller 
    {
    public:
      inline void operator() ( T*& object ) const
      {
        object->differentiate();
      }
    };
  
    
    for_each( theReactorVector.begin(), theReactorVector.end(), 
  	      DifferentiateCaller<Reactor>() );


    (Kouichi Takahashi)
  */

#define FOR_ALL( SEQCLASS, SEQ, METHOD )\
  for( SEQCLASS ## ConstIterator i( (SEQ) .begin() ) ;\
       i != (SEQ) .end() ; ++i )\
    { (*i)-> METHOD (); }

 

} // namespace libecs


#endif /* ___UTIL_H___ */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/

