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


#ifndef ___DEFS_H___
#define ___DEFS_H___
#include <sys/types.h>
#include <float.h>
#include "config.h"

// system constants

const int RANDOM_NUMBER_BUFFER_SIZE(65535);
const char* const REACTOR_SO_DIR= DM_SO_DIR "/reactor"; 

// FIXME: use numeric_limits
#if defined(sparc) && defined(__SVR4)
// SPARC SUN Solaris
#define QUANTITY_MAX LONG_LONG_MAX
#define QUANTITY_MIN LONG_LONG_MIN
typedef long long Int;
typedef long double Float;
typedef long double Mol;
#if (defined(__GNUC__))
#define FLOAT_DIG DBL_DIG
#else // (defined(__GNUC__))
#define FLOAT_DIG 15
#endif // (defined(__GNUC__))
#include <math.h>
inline Float MODF(Float x, Float *y) {
  double frac_part, int_part;
  frac_part = modf(static_cast<double>(x), &int_part);
  *y = int_part;
  return frac_part;
}
#elif (defined(__linux__) && defined(__alpha__))
// long = long long = double = long double = 64 bit
#define QUANTITY_MAX LONG_MAX
#define QUANTITY_MIN LONG_MIN
typedef long Int;
typedef double Float;
typedef Float Mol;
#define MODF modf
#define DOUBLE_FLOAT
#if (defined(__GNUC__))
#define FLOAT_DIG DBL_DIG
#else // (defined(__GNUC__))
#define FLOAT_DIG 15
#endif // (defined(__GNUC__))
#else  
// typedef Int as 64 bit integer, Float and Mol as long double.
#define QUANTITY_MAX LONG_LONG_MAX
#define QUANTITY_MIN LONG_LONG_MIN
typedef int64_t Int;
typedef long double Float;
typedef long double Mol;
#define MODF modfl
#define LONG_DOUBLE_FLOAT
#if (defined(__GNUC__))
#define FLOAT_DIG LDBL_DIG
#else // (defined(__GNUC__))
#define FLOAT_DIG 18  // expecting long double to be 96 bit numbers...
#endif // (defined(__GNUC__))
#endif 

typedef Int Quantity;
typedef Float Concentration;


//! Avogadro number. 
const Float N_A = 6.0221367e+23;

const char FIELD_SEPARATOR = ' ';
const char GROUP_SEPARATOR = '\t';
const char LINE_SEPARATOR = '\n';


const int NOMATCH = -1;




#endif /* ___DEFS_H___ */





