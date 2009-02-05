//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2009 Keio University
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
#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include <limits>
#include <time.h>

#include "Exceptions.hpp"
#include "Util.hpp"
#include "Polymorph.hpp"

namespace libecs
{

#define __STRINGCAST_SPECIALIZATION_DEF( NEW, GIVEN )\
  template<> const NEW stringCast<NEW,GIVEN>( const GIVEN& aValue )\
  {\
    return boost::lexical_cast<NEW>( aValue );\
  } //

  __STRINGCAST_SPECIALIZATION_DEF( String, Real );
#if !HIGHREAL_IS_REAL
  __STRINGCAST_SPECIALIZATION_DEF( String, HighReal );
#endif
  __STRINGCAST_SPECIALIZATION_DEF( String, Integer );
  __STRINGCAST_SPECIALIZATION_DEF( String, UnsignedInteger );

  __STRINGCAST_SPECIALIZATION_DEF( Integer, String );
  __STRINGCAST_SPECIALIZATION_DEF( UnsignedInteger, String );
  // __STRINGCAST_SPECIALIZATION_DEF( String, String );


  // boost::lexical_cast does not convert Inf and NaN.
  // Specialization here for <Real,String> does this job.

  /* original:
  __STRINGCAST_SPECIALIZATION_DEF( Real, String );
#i !HIGHREAL_IS_REAL
  __STRINGCAST_SPECIALIZATION_DEF( HighReal, String );
#endif
  */

  template< typename T >
  const Real stringToFloat( StringCref aValue )
  {
    try
      {
	return boost::lexical_cast<Real>( aValue );
      }
    catch( const boost::bad_lexical_cast& )
      {
	String aCaseless( aValue );
	std::transform( aCaseless.begin(), aCaseless.end(), 
			aCaseless.begin(), 
			tolower );
	
	if( aCaseless == "inf" || aCaseless == "infinity" )
	  {
	    return std::numeric_limits<Real>::infinity();
	  }
	else if( aCaseless.substr( 0, 3 ) == "nan" )
	  {
	    return std::numeric_limits<Real>::quiet_NaN();
	  }

	throw;
      }
  }

  template<>
  const Real stringCast<Real,String>( StringCref aValue )
  {
    return stringToFloat<Real>( aValue );
  }

#if !HIGHREAL_IS_REAL
  template<>
  const HighReal stringCast<HighReal,String>( StringCref aValue )
  {
    return stringToFloat<HighReal>( aValue );
  }
#endif


#undef __STRINGCAST_SPECIALIZATION_DEF


  void eraseWhiteSpaces( StringRef str )
  {
    static const String aSpaceCharacters( " \t\n" );

    String::size_type p( str.find_first_of( aSpaceCharacters ) );

    while( p != String::npos )
      {
	str.erase( p, 1 );
	p = str.find_first_of( aSpaceCharacters, p );
      }
  }

  void throwSequenceSizeError( const size_t aSize, 
			       const size_t aMin, const size_t aMax )
  {
    THROW_EXCEPTION( RangeError,
		     "Size of the sequence must be within [ " 
		     + stringCast( aMin ) + ", " + stringCast( aMax )
		     + " ] ( " + stringCast( aSize ) + " given)." );
  }

  void throwSequenceSizeError( const size_t aSize, const size_t aMin )
  {
    THROW_EXCEPTION( RangeError,
		     "Size of the sequence must be at least " 
		     + stringCast( aMin ) + 
		     + " ( " + stringCast( aSize ) + " given)." );
  }



  const Polymorph convertStringMapToPolymorph( StringMap const& aMap )
  {
    PolymorphVector aVector;
    aVector.reserve( aMap.size() );

    for( StringMap::const_iterator i( aMap.begin() ); 
	 i != aMap.end();  ++i )
      {
	PolymorphVector anInnerVector;
	anInnerVector.push_back( i->first );
	anInnerVector.push_back( i->second );

	aVector.push_back( anInnerVector );
      }

    return aVector;
  }





} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
