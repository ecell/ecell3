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

#include <limits>
#include <time.h>

#include "Exceptions.hpp"
#include "Util.hpp"
#include "Polymorph.hpp"

namespace libecs
{


  template<> const Real stringTo<Real>( StringCref str )
  {
    // FIXME: error check, throw exception
    return strtod( str.c_str(), NULLPTR );
  }

  template<> const Integer stringTo<Integer>( StringCref str )
  {
    // FIXME: error check, throw exception
    return strtol( str.c_str(), NULLPTR, 10 );
  }

  template<> const UnsignedInteger stringTo<UnsignedInteger>( StringCref str )
  {
    // FIXME: error check, throw exception
    return strtoul( str.c_str(), NULLPTR, 10 );
  }

  template <> inline const String toString( const Real& t )
  {
    return boost::lexical_cast<String>( t );
  }

  template <> inline const String toString( const Integer& t )
  {
    return boost::lexical_cast<String>( t );
  }

  template <> inline const String toString( const UnsignedInteger& t )
  {
    return boost::lexical_cast<String>( t );
  }

  template <> inline const String toString( const String& t )
  {
    return t;
  }

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

  void throwSequenceSizeError( const int aSize, 
			       const int aMin, const int aMax )
  {
    THROW_EXCEPTION( RangeError,
		     "Size of the sequence must be within [ " 
		     + toString( aMin ) + ", " + toString( aMax )
		     + " ] ( " + toString( aSize ) + " given)." );
  }

  void throwSequenceSizeError( const int aSize, const int aMin )
  {
    THROW_EXCEPTION( RangeError,
		     "Size of the sequence must be at least " 
		     + toString( aMin ) + 
		     + " ( " + toString( aSize ) + " given)." );
  }



  const Polymorph convertStringMapToPolymorph( StringMapCref aMap )
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


#ifdef UTIL_TEST

#include <iostream>

using namespace std;
using namespace libecs;

main()
{
  String str( "  \t  a bcde f\tghi\n\t jkl\n \tmnopq     \n   \t " );
  eraseWhiteSpaces( str );
  cerr << '[' << str << ']' << endl;

}

#endif /* UTIL_TEST */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
