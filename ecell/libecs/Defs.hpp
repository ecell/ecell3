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


#ifndef __DEFS_HPP
#define __DEFS_HPP

#include <stdint.h>
#include <float.h>
#include <string>
#include <list>
#include <vector>

#include "ecell/config.h"

#include "CoreLinuxCompatibility.hpp"
#include "AssocVector.h"



#define DEBUG


// stringifiers.  see preprocessor manual
#define XSTR( S ) STR( S )
#define STR( S ) #S


#define USE_LIBECS using namespace libecs

// cmath

#if defined( HAVE_CMATH )
#include <cmath>
#elif defined( HAVE_MATH )
#include <math>
#else
#error "either math or cmath header is needed."
#endif /* HAVE_CMATH */

namespace libecs
{

  // system constants

  const int MAJOR_VERSION( ECELL_MAJOR_VERSION );
  const int MINOR_VERSION( ECELL_MINOR_VERSION );
  const int MICRO_VERSION( ECELL_MICRO_VERSION );

  const char* const VERSION_STRING( ECELL_VERSION_STRING );


  inline const int getMajorVersion()
  {
    return MAJOR_VERSION;
  }

  inline const int getMinorVersion()
  {
    return MINOR_VERSION;
  }

  inline const int getMicroVersion()
  {
    return MICRO_VERSION;
  }

  inline const std::string getVersion()
  {
    return VERSION_STRING;
  }



  // CoreLinux++ compatibility

  using namespace corelinux;

  // replace CORELINUX from macro names

#define DECLARE_LIST        CORELINUX_LIST
#define DECLARE_VECTOR      CORELINUX_VECTOR
#define DECLARE_MAP         CORELINUX_MAP
#define DECLARE_MULTIMAP    CORELINUX_MULTIMAP  
#define DECLARE_SET         CORELINUX_SET       
#define DECLARE_MULTISET    CORELINUX_MULTISET  
#define DECLARE_QUEUE       CORELINUX_QUEUE     
#define DECLARE_STACK       CORELINUX_STACK     


  
  // from Loki

  
#define DECLARE_ASSOCVECTOR(key,value,comp,name)                             \
      typedef ::Loki::AssocVector<key,value,comp > name;                      \
      typedef name *       name ## Ptr;                            \
      typedef const name * name ## Cptr;                           \
      typedef name &       name ## Ref;                            \
      typedef const name & name ## Cref;                           \
      typedef name::iterator name ## Iterator;                     \
      typedef name::iterator& name ## IteratorRef;                 \
      typedef name::iterator* name ## IteratorPtr;                 \
      typedef name::const_iterator name ## ConstIterator;          \
      typedef name::const_iterator& name ## ConstIteratorRef;      \
      typedef name::const_iterator* name ## ConstIteratorPtr;      \
      typedef name::reverse_iterator name ## Riterator;            \
      typedef name::reverse_iterator& name ## RiteratorRef;        \
      typedef name::reverse_iterator* name ## RiteratorPtr
   


  // String

  DECLARE_TYPE( std::string, String );

  DECLARE_TYPE( const char* const, StringLiteral );

  // Numeric types

  DECLARE_TYPE( long int, Int );
  DECLARE_TYPE( unsigned long int, UnsignedInt );

  // these can cause problem when used as template parameters
  //  DECLARE_TYPE( int64_t, Int );
  //  DECLARE_TYPE( uint64_t, UnsignedInt );

  DECLARE_TYPE( double, Real );


  //! Avogadro number. 
  const Real N_A( 6.0221367e+23 );


  // MACROS

#if 0

#if !defined( HAVE_PRETTY_FUNCTION )
#define __PRETTY_FUNCTION__ ""
#endif

#endif // 0

  /**
     Converts each type into a unique, insipid type.
     Invocation Type2Type<T> where T is a type.
     Defines the type OriginalType which maps back to T.
     
     taken from loki library.

     @ingroup util
  */

  template <typename T>
  struct Type2Type
  {
    typedef T OriginalType;
  };


} // namespace libecs


#endif /* __DEFS_HPP */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/



