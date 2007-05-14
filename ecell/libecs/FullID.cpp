//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#include <string>

#include "Util.hpp"
#include "Exceptions.hpp"

#include "FullID.hpp"

namespace libecs
{

  ///////////////////////  SystemPath

  void SystemPath::parse( StringCref systempathstring )
  {
    if( systempathstring.empty() )
      {
	return;
      }

    String aString( systempathstring );
    eraseWhiteSpaces( aString );
    
    String::size_type aFieldStart( 0 );
    
    // absolute path ( start with '/' )
    if( aString[0] == DELIMITER )
       {
	 //insert(end(), String( 1, DELIMITER ) );
	 push_back( String( 1, DELIMITER ) );

	 if( aString.size() == 1 )
	   {
	     return;
	   }

	 ++aFieldStart;
       }

    String::size_type aFieldEnd( aString.find_first_of( DELIMITER, 
							aFieldStart ) );
    //    insert(end(), aString.substr( aFieldStart, 
    //			       aFieldEnd - aFieldStart ) );
    push_back( aString.substr( aFieldStart, 
			       aFieldEnd - aFieldStart ) );

    while( aFieldEnd != String::npos  )
      {
	aFieldStart = aFieldEnd + 1;
	aFieldEnd = aString.find_first_of( DELIMITER, aFieldStart );
	
	insert(end(), aString.substr( aFieldStart, 
				      aFieldEnd - aFieldStart ) );
      }

  }

  const String SystemPath::getString() const
  {
    StringList::const_iterator i = begin();
    String aString;

    if( isAbsolute() )
      {
	if( size() == 1 )
	  {
	    return "/";
	  }
	else
	  {
	    ; // do nothing
	  }
      }
    else
      {
	// isAbsolute() == false implies that this can be empty
	if( empty() )
	  {
	    return aString;
	  }
	else
	  {
	    aString = *i;
	  }
      }

    if( i == end() ) {
        return aString;
    }

    ++i;

    while( i != end() )
      {
	aString += '/';
	aString += *i;
	++i;
      }

    return aString;
  }



  ///////////////// FullID

  void FullID::parse( StringCref fullidstring )
  {
    // empty FullID string is invalid
    if( fullidstring == "" )
      {
	THROW_EXCEPTION( BadID, "Empty FullID string." );
      }

    String aString( fullidstring );
    eraseWhiteSpaces( aString );

    // ignore leading white spaces
    String::size_type aFieldStart( 0 );
    String::size_type aFieldEnd( aString.find_first_of( DELIMITER,
							aFieldStart ) );
    if( aFieldEnd == String::npos )
      {
	THROW_EXCEPTION( BadID, 
			 "No ':' in the FullID string [" + aString + "]." );
      }

    String aTypeString( aString.substr( aFieldStart, 
					aFieldEnd - aFieldStart ) );
    theEntityType = EntityType( aTypeString );
    
    aFieldStart = aFieldEnd + 1;
    aFieldEnd = aString.find_first_of( DELIMITER, aFieldStart );
    if( aFieldEnd == String::npos )
      {
	THROW_EXCEPTION( BadID, 
			 "Only one ':' in the FullID string [" 
			 + aString + "]." );
      }

    theSystemPath = 
      SystemPath( aString.substr( aFieldStart, 
				  aFieldEnd - aFieldStart ) );
    
    aFieldStart = aFieldEnd + 1;

    // drop trailing string after extra ':'(if this is  FullPN),
    // or go to the end
    aFieldEnd = aString.find_first_of( DELIMITER, aFieldStart );

    theID = aString.substr( aFieldStart, aFieldEnd - aFieldStart );
  }    

  const String FullID::getString() const
  {
    return theEntityType.getString() + FullID::DELIMITER 
      + theSystemPath.getString() + FullID::DELIMITER + theID;
  }

  bool FullID::isValid() const
  {
    bool aFlag( theSystemPath.isValid() );
    aFlag &= ! theID.empty();

    return aFlag;
  }


  ///////////////// FullPN


  FullPN::FullPN( StringCref fullpropertynamestring )
    :
    theFullID( fullpropertynamestring )
  {

    String::size_type aPosition( 0 );

    for( int i( 0 ) ; i < 3 ; ++i )
      {
	aPosition = fullpropertynamestring.
	  find_first_of( FullID::DELIMITER, aPosition );
	if( aPosition == String::npos ) 
	  {
	    THROW_EXCEPTION( BadID,
			     "Not enough fields in FullPN string [" +
			     fullpropertynamestring + "]." );
	  }
	++aPosition;
      }

    thePropertyName = fullpropertynamestring.substr( aPosition, String::npos );
    eraseWhiteSpaces( thePropertyName );
  }

  const String FullPN::getString() const
  {
    return theFullID.getString() + FullID::DELIMITER + thePropertyName;
  }

  bool FullPN::isValid() const
  {
    return theFullID.isValid() & ! thePropertyName.empty();
  }

} // namespace libecs

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
