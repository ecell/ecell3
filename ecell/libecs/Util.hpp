//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
// 		This file is part of Serizawa (E-CELL Core System)
//
//	       written by Kouichi Takahashi  <shafi@sfc.keio.ac.jp>
//
//                              E-CELL Project,
//                          Lab. for Bioinformatics,  
//                             Keio University.
//
//             (see http://www.e-cell.org for details about E-CELL)
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// Serizawa is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// Serizawa is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with Serizawa -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER





#ifndef ___UTIL_H___
#define ___UTIL_H___
#include <string>
#include <typeinfo>
#include "include/Defs.h"
#include "korandom/korandom.h"

/////////////// KoRandom random number generator

#if defined(DOUBLE_FLOAT)        // Float is typedef'd as double
typedef korandom_d_c  RandomNumberGenerator;
#elif defined(LONG_DOUBLE_FLOAT) // Float is long double
typedef korandom_ld_c RandomNumberGenerator; 
#else                            // fail-safe
typedef korandom_d_c  RandomNumberGenerator;
#endif // DOUBLE_FLOAT

extern RandomNumberGenerator* theRandomNumberGenerator;

/////////////// least common multiple

inline int lcm(int a, int b)
{
  if(a>b)
    {
      int i;
      for(i=1;(a*i)%b;++i) ;
      return a*i;
    }
  else
    {
      int i;
      for(i=1;(b*i)%a;++i) ;
      return b*i;
    }
}

////////////////////// table_lookup

int table_lookup(const string& str,const char** table);

//////////////////////  string -> Float,Int

Float asFloat(const string& str);

Int asInt(const string& str);

//////////////////////  extract just a filename from a path

string basenameOf(const string& str, string::size_type maxlength = 0);


//////////////////////  reversed order compare

template <class T>
class ReverseCmp
{
public:
  bool operator()(const T x, const T y) const {return x > y;}
};

#endif /* ___UTIL_H___ */
