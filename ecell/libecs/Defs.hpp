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


#ifndef ___DEFS_H___
#define ___DEFS_H___
#include <stdint.h>
#include <float.h>
#include <stl.h>
#include "config.h"

// system constants

const int RANDOM_NUMBER_BUFFER_SIZE(65535);


// Numeric types

// FIXME: use numeric_limits
typedef int64_t Int;
typedef uint64_t Uint;
const int INT_SIZE=sizeof(int64_t);

typedef double Float;
typedef Float Mol;
#define FLOAT_DIG DBL_DIG
typedef Int Quantity;
typedef Float Concentration;


//! Avogadro number. 
const Float N_A = 6.0221367e+23;

const char FIELD_SEPARATOR = ' ';
const char GROUP_SEPARATOR = '\t';
const char LINE_SEPARATOR = '\n';


const int NOMATCH = -1;

#define ATOF( str ) atof( str )
#define ATOI( str ) atoi( str )




// CoreLinux++ compatibility

#define DECLARE_TYPE( mydecl, mytype )  \
typedef mydecl         mytype;         \
typedef mytype *       mytype ## Ptr;  \
typedef const mytype * mytype ## Cptr; \
typedef mytype &       mytype ## Ref;  \
typedef const mytype & mytype ## Cref;


#define DECLARE_CLASS( tag )            \
   class   tag;                        \
   typedef tag *       tag ## Ptr;     \
   typedef const tag * tag ## Cptr;    \
   typedef tag &       tag ## Ref;     \
   typedef const tag & tag ## Cref;

#define NULLPTR 0


// String and StringPair

#include <string>

DECLARE_TYPE( string, String );
typedef pair<String,String> StringPair_;
DECLARE_TYPE( StringPair_, StringPair );


#endif /* ___DEFS_H___ */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/



