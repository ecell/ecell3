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
#include <string>
#include <typeinfo>
#include "Defs.hpp"
#include "korandom/korandom.h"

typedef korandom_d_c  RandomNumberGenerator;

/**
   Random number generator
 */
//FIXME: thread safe?
extern RandomNumberGenerator* theRandomNumberGenerator; 

/**
   least common multiple.
 */

inline int lcm( int a, int b )
{
  if( a > b )
    {
      int i;
      for( i = 1; ( a * i ) % b ; ++i ) 
	{
	  ; // do nothing
	}
      return a * i;
    }
  else
    {
      int i;
      for( i = 1 ; ( b * i ) % a ; ++i ) 
	{
	  ; // do nothing
	}
      return b * i;
    }
}

/**
   table lookup function.
 */

int table_lookup( StringCref str, const char** table );

/** 
    universal String -> object converter.
    Float and Int specializations are defined in Util.cpp.
    Conversion to the other classes are conducted using 
    istrstream.
 */

template<class T> T stringTo( StringCref str );

/**
   extract a filename from a path string
*/

String basenameOf( StringCref str, String::size_type maxlength = 0 );


/**
   reversed order compare
*/

template <class T>
class ReverseCmp
{
public:
  bool operator()( const T x, const T y ) const { return x > y; }
};

#endif /* ___UTIL_H___ */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/

