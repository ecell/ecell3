//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2008 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell System is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
//
// E-Cell System is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public
// License along with E-Cell System -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
//
//END_HEADER
//
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#ifndef __UTIL_HPP
#define __UTIL_HPP

#include <stdlib.h>
#include <sstream>
#include <functional>
#include <limits>

#include "libecs.hpp"
#include "Exceptions.hpp"

/**
   @addtogroup util The Utilities.
   Utilities.

   @ingroup libecs
   @{
 */

/** @file */


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

   @note This will be deprecated.  Use select2nd instead.

   @arg SEQCLASS the classname of the STL sequence.
   @arg SEQ the STL sequence.
   @arg METHOD the name of the method.
  
   @see FOR_ALL
*/
#define FOR_ALL_SECOND( SEQCLASS, SEQ, METHOD )\
  FOR_ALL( SEQCLASS, SEQ )\
    { (*i).second-> METHOD (); }

namespace libecs {

template< class T >
struct PtrGreater
{
    bool operator()( T x, T y ) const {
        return *y < *x;
    }
};

template< class T >
struct PtrLess
{
    bool operator()( T x, T y ) const {
        return *y > *x;
    }
};

/**
   Check if aSequence's size() is within [ aMin, aMax ].

   If not, throw an OutOfRange exception.

*/
template <class Sequence>
void checkSequenceSize( const Sequence& aSequence,
                        const typename Sequence::size_type aMin,
                        const typename Sequence::size_type aMax )
{
    const typename Sequence::size_type aSize( aSequence.size() );
    if ( aSize < aMin || aSize > aMax )
    {
        throwSequenceSizeError( aSize, aMin, aMax );
    }
}

/**
   Check if aSequence's size() is at least aMin.

   If not, throw an OutOfRange exception.
*/
template <class Sequence>
void checkSequenceSize( const Sequence& aSequence,
                        const typename Sequence::size_type aMin )
{
    const typename Sequence::size_type aSize( aSequence.size() );
    if ( aSize < aMin )
    {
        throwSequenceSizeError( aSize, aMin );
    }
}

///@internal
LIBECS_API void throwSequenceSizeError( const size_t aSize,
                                        const size_t aMin, const size_t aMax );

///@internal
LIBECS_API void throwSequenceSizeError( const size_t aSize, const size_t aMin );

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

/**
   Retrieves the temporary directory from the system settings.
 */
LIBECS_API const char* getTempDirectory();

/**
   Erase white space characters ( ' ', '\t', and '\n' ) from a string
*/
LIBECS_API void eraseWhiteSpaces( StringRef str );


} // namespace libecs

/** @} */

#endif /* __UTIL_HPP */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/

