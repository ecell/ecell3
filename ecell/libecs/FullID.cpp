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
    StringListConstIterator i = begin();
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
	throw BadID( __PRETTY_FUNCTION__,
		     "Empty FullID string." );
      }

    String aString( fullidstring );
    eraseWhiteSpaces( aString );

    // ignore leading white spaces
    String::size_type aFieldStart( 0 );
    String::size_type aFieldEnd( aString.find_first_of( DELIMITER,
							aFieldStart ) );
    if( aFieldEnd == String::npos )
      {
	throw BadID( __PRETTY_FUNCTION__,
		     "No ':' in the FullID string [" + aString + "]." );
      }

    String aTypeString( aString.substr( aFieldStart, 
					aFieldEnd - aFieldStart ) );
    thePrimitiveType = PrimitiveType( aTypeString );
    
    aFieldStart = aFieldEnd + 1;
    aFieldEnd = aString.find_first_of( DELIMITER, aFieldStart );
    if( aFieldEnd == String::npos )
      {
	throw BadID( __PRETTY_FUNCTION__,
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
    return thePrimitiveType.getString() + FullID::DELIMITER 
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

    for( Int i( 0 ) ; i < 3 ; ++i )
      {
	aPosition = fullpropertynamestring.
	  find_first_of( FullID::DELIMITER, aPosition );
	if( aPosition == String::npos ) 
	  {
	    throw BadID( __PRETTY_FUNCTION__, 
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

#ifdef TEST_FULLID

using namespace libecs;

main()
{
  SystemPath aSystemPath( "   \t  /A/BB/CCC//DDDD/EEEEEE    \t \n  " );
  cout << aSystemPath.getString() << endl;

  SystemPath aSystemPath2( aSystemPath );
  cout << aSystemPath2.getString() << endl;

  aSystemPath2.pop_front();
  aSystemPath2.pop_back();
  cout << aSystemPath2.getString() << endl;

  SystemPath aSystemPath3( "/" );
  cout << aSystemPath3.getString() << endl;
  cout << aSystemPath3.size() << endl;
  aSystemPath3.pop_front();
  cout << aSystemPath3.getString() << endl;
  cout << aSystemPath3.size() << endl;
  cout << aSystemPath3.empty() << endl;

  while( aSystemPath.size() != 0 )
    {
      cout << aSystemPath.size() << " : " << aSystemPath.isAbsolute() << " : " 
	   << aSystemPath.getString() << endl;
      aSystemPath.pop_front();
    }

  //  SystemPath aSystemPath2( "/A/../B" );
  //  cout << aSystemPath2.getString() << endl;

  SystemPath aSystemPath4( "/CYTOPLASM" );
  cout << aSystemPath4.getString() << endl;

  cout << "\n::::::::::" << endl;

  try
    {
      FullID aFullID( "       \t  \n  Substance:/A/B:S   \t   \n" );
      cout << aFullID.getString() << endl;
      cout << aFullID.getPrimitiveType() << endl;
      cout << aFullID.getSystemPath().getString() << endl;
      cout << aFullID.getID() << endl;
      cout << aFullID.isValid() << endl;

      FullID aFullID2( aFullID );
      cout << aFullID2.getString() << endl;

      FullID aFullID3( "Reactor:/:R" );
      cout << aFullID3.getString() << endl;
      aFullID3 = aFullID2;
      cout << aFullID3.getString() << endl;

      cout << "\n::::::::::" << endl;
      //      FullPN aFullPN( 1,aFullID.getSystemPath(),"/", "PNAME" );

      FullPN 
	aFullPN( "       \t  \n  Substance:/A/B:S:PNAME   \t   \n" );
      cout << aFullPN.getString() << endl;
      cout << aFullPN.getPrimitiveType() << endl;
      cout << aFullPN.getSystemPath().getString() << endl;
      cout << aFullPN.getID() << endl;
      cout << aFullPN.getPropertyName() << endl;
      cout << aFullPN.isValid() << endl;

      FullPN aFullPN2( aFullPN );
      cout << aFullPN2.getString() << endl;

      FullPN aFullPN3( "Reactor:/:R:P" );
      cout << aFullPN3.getString() << endl;
      aFullPN3 = aFullPN2;
      cout << aFullPN3.getString() << endl;

    }
  catch ( ExceptionCref e )
    {
      cerr << e.message() << endl;
    }

}

// g++ -I.. -I../.. FullID.cpp PrimitiveType.cpp Util.cpp -DTEST_FULLID ../../korandom/.libs/libkorandom.a 

#endif /* TEST_FULLID */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
