//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 2001 Keio University
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

#ifndef __RCPTR_HPP
#define __RCPTR_HPP

#include "Defs.hpp"


/** @addtogroup util
 *@{
 */

/** @file */


/**
   Use this macro if you want to use 
   SomeClassRCPtr instead of RCPtr<SomeClass>.

*/

#define DECLARE_RCPTR( type )\
DECLARE_TYPE( RCPtr<type>, type ## RCPtr )


/**
   A simple reference counted pointer class.

   Inspired by, but rewritten version of, the AutoRelease Library
   (Reference Counting Garbage Collector for C++) taken from
   http://www.fukt.hk-r.se/~per/autorelease/
   written by Per Liden <per@fukt.hk-r.se>.

*/


template <typename T>
class RCPtr
{

  DECLARE_TYPE( RCPtr<T>, TRCPtr );

public:

  RCPtr()
    :
    theObject( NULLPTR ),
    theCount( NULLPTR )
  {
    ; // do nothing
  }

  RCPtr( TRCPtrCref rhs )
  {
    if( rhs.isNonNull() )
      {
	theObject = rhs.theObject;
	theCount  = rhs.theCount;
	incrementReferenceCount();
      }
    else
      {
	theObject = NULLPTR;
	theCount  = NULLPTR;
      }
  }

  explicit RCPtr( const T* rhs )
  {
    if( rhs != NULLPTR )
      {
	theObject = const_cast<T*>( rhs );
	theCount  = new unsigned int( 1 );
      }
    else
      {
	theObject = NULLPTR;
	theCount  = NULLPTR;
      }
  }

  explicit RCPtr( T& rhs )
    :
    theObject( &rhs ),
    theCount( new unsigned int( 1 ) )
  {
    ; // do nothing
  }

  ~RCPtr()
  {
    if( isNonNull() )
      {
	decrementReferenceCount();
      }
  }

  TRCPtrRef operator =( TRCPtrCref rhs )
  {
    if( rhs.isNonNull() )
      {
	rhs.incrementReferenceCount();
      }
 
    if( isNonNull() )
      {
	decrementReferenceCount();
      }

    theObject = rhs.theObject;
    theCount  = rhs.theCount;

    return *this;
  }

  TRCPtrRef operator =( const T* rhs )
  {
    if( isNonNull() )
      {
	decrementReferenceCount();
      }

    if( rhs != NULLPTR )
      {
	theObject = const_cast<T*>( rhs );
	theCount  = new unsigned int( 1 );
      }
    else
      {
	theObject = NULLPTR;
	theCount  = NULLPTR;
      }

    return *this;
  }

  T* operator ->() const 
  {
    return theObject;
  }

  T& operator *() const
  {
    return *theObject;
  }

  operator T*() const
  {
    return theObject;
  }

  operator T() const
  {
    return *theObject;
  }

private:

  const bool isNonNull() const
  {
    return theObject != NULLPTR;
  }

  void incrementReferenceCount() const
  {
    ++(*theCount);
  }

  void decrementReferenceCount() const
  {
    if( (*theCount) == 1 )
      {
	delete theCount;
	delete theObject;
      }
    else
      {
	--(*theCount);
      }
  }

private:

  T*                    theObject;
  mutable unsigned int* theCount;

};

//@}
  


#endif /* __RCPTR_HPP */
