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

#include <time.h>

#include "Exceptions.hpp"
#include "Util.hpp"


RandomNumberGenerator* theRandomNumberGenerator = 
new RandomNumberGenerator( 
			  // FIXME: is this cast good?
			  // should be reinterpret_cast or something?
			  (Float)(time(NULL)) ,
			  RANDOM_NUMBER_BUFFER_SIZE);

int table_lookup( StringCref str, const char** table )
{
  for( int i = 0 ; table[i] != NULL ; ++i )
    {
      if( str == String( table[ i ] ) )
	{
	  return i;
	}
    }

  throw NotFound( __PRETTY_FUNCTION__ );
}

template<> const Float stringTo<Float>( StringCref str )
{
  // FIXME: error check, throw exception
  return strtod( str.c_str(), NULL );
}

template<> const Int stringTo<Int>( StringCref str )
{
  // FIXME: error check, throw exception
  return strtol( str.c_str(), NULL, 10 );
}

template<> const UnsignedInt stringTo<UnsignedInt>( StringCref str )
{
  // FIXME: error check, throw exception
  return strtoul( str.c_str(), NULL, 10 );
}

template<> const String toString<Float>( const Float& f )
{ 
  ostrstream os;
  os.precision( FLOAT_DIG );
  os << f;
  os << ends;
  return os.str();
}

string basenameOf( StringCref str, String::size_type maxlength )
{
  String::size_type s = str.rfind( '/' );
  if( s == String::npos )
    {
      s = 0;
    } 
  else
    {
      s++;
    }

  String::size_type e = str.rfind( '.' );
  if( e == String::npos )
    {
      e = str.size();
    }

  String::size_type l = e - s;
  if( maxlength != 0 && maxlength < l ) 
    {
      l = maxlength;
    }
  return str.substr( s, l );
}


#ifdef UTIL_TEST

main()
{


}

#endif /* UTIL_TEST */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
