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

#include <strstream>
#include "Message.hpp"
#include "StringList.hpp"

////////////////////// Message

Message::Message( StringCref keyword, StringCref body ) 
  :
  StringPair( keyword, body ) 
{
  ; // do nothing

}

Message::Message( StringCref message ) 
{
  String::size_type j = message.find( FIELD_SEPARATOR );
  if( j != String::npos )
    {
      first = message.substr( 0, j );
      String::size_type k = message.find_first_not_of( FIELD_SEPARATOR, j );
      second = message.substr( k, String::npos );
    }
  else
    {
      first = message;
      second = "";
    }
}

Message::Message( StringCref keyword, const Float f )
{
  first = keyword;
  ostrstream os;
  os.precision( FLOAT_DIG );
  os << f;
  os << ends;  // by naota on 29. Nov. 1999
  second = os.str();
}

Message::Message( StringCref keyword, const Int i )
{
  first = keyword;
  ostrstream os;
  os << i;
  os << ends;  // by naota on 29. Nov. 1999
  second = os.str();
}

Message::Message( MessageCref message )
  :
  StringPair( message.getKeyword(), message.getBody() )
{
  ; // do nothing
}

Message& Message::operator=( MessageCref rhs )
{
  if( this != &rhs )
    {
      first = rhs.getKeyword();
      second = rhs.getBody();
    }

  return *this;
}

Message::~Message()
{
  ; // do nothing
}

const String Message::getBody( int n ) const
{
  String::size_type pos( 0 );
  while( n != 0 )
    {
      pos = second.find( FIELD_SEPARATOR, pos );
      if( pos == String::npos )
	return "";
      ++pos;
      --n;
    }
  return second.substr( pos, second.find( FIELD_SEPARATOR ) - pos );
}



/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
