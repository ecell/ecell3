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

const int RANDOM_NUMBER_BUFFER_SIZE( 65535 );


// CoreLinux++ compatibility

#include "CoreLinuxCompatibility.hpp"


// replace CORELINUX from macro names

#define DECLARE_LIST        CORELINUX_LIST
#define DECLARE_VECTOR      CORELINUX_VECTOR
#define DECLARE_MAP         CORELINUX_MAP       
#define DECLARE_MULTIMAP    CORELINUX_MULTIMAP  
#define DECLARE_SET         CORELINUX_SET       
#define DECLARE_MULTISET    CORELINUX_MULTISET  
#define DECLARE_QUEUE       CORELINUX_QUEUE     
#define DECLARE_STACK       CORELINUX_STACK     



// String

#include <string>

DECLARE_TYPE( std::string, String );


// Numeric types

// FIXME: use numeric_limits
DECLARE_TYPE( int64_t, Int );
DECLARE_TYPE( uint64_t, UnsignedInt );
const int INT_SIZE( sizeof( Int ) );

DECLARE_TYPE( double, Real );
const int FLOAT_DIG( DBL_DIG );

//! Avogadro number. 
const Real N_A = 6.0221367e+23;

const int NOMATCH = -1;

#endif /* ___DEFS_H___ */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/



