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
//FIXME: const char* const REACTOR_SO_DIR= DM_SO_DIR "/reactor"; 

// FIXME: use numeric_limits
#if SIZEOF_LONG_LONG > 0
typedef long long Int;
const int INT_SIZE=SIZEOF_LONG_LONG;
#else
typedef long Int;
const int INT_SIZE=SIZEOF_LONG;
#endif

//#define QUANTITY_MAX LONG_LONG_MAX
//#define QUANTITY_MIN LONG_LONG_MIN

#ifdef HAVE_LONG_DOUBLE
  typedef long double Float;
  typedef long double Mol;
#  if (defined(__GNUC__))
#    define FLOAT_DIG LDBL_DIG
#  else // (defined(__GNUC__))
#    define FLOAT_DIG 18  // expecting long double to be 96 bit numbers...
#  endif // (defined(__GNUC__))
#  define MODF modfl
#else /* HAVE_LONG_DOUBLE */
  typedef double Float;
  typedef double Mol;
#  if (defined(__GNUC__))
#    define FLOAT_DIG DBL_DIG
#  else // (defined(__GNUC__))
#    define FLOAT_DIG 15
#  endif // (defined(__GNUC__))
#  define MODF modf
#endif /* HAVE_LONG_DOUBLE */

//double modf(double,double*);
extern "C"{
long double modfl(long double,long double*);
}

#if defined(sparc) && defined(__SVR4)
#include <math.h>
#undef MODF
inline Float MODF(Float x, Float *y) {
  double frac_part, int_part;
  frac_part = modf(static_cast<double>(x), &int_part);
  *y = int_part;
  return frac_part;
}
#endif /* defined(sparc) && defined(__SVR4) */


typedef Int Quantity;
typedef Float Concentration;


//! Avogadro number. 
const Float N_A = 6.0221367e+23;

const char FIELD_SEPARATOR = ' ';
const char GROUP_SEPARATOR = '\t';
const char LINE_SEPARATOR = '\n';


const int NOMATCH = -1;




#endif /* ___DEFS_H___ */





