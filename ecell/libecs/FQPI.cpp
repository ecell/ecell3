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

#include <string>
#include "FQPI.hpp"

namespace libecs
{

  ///////////////////////  SystemPath

  SystemPath::SystemPath( StringCref rqsn ) 
    :
    theSystemPath( rqsn )
  {
    standardize();
  }

  SystemPath::SystemPath( SystemPathCref systempath )
    :
    theSystemPath( systempath.getSystemPathString() )
  {
    standardize();
  }

  const String SystemPath::last() const
  {
    int anInt( theSystemPath.rfind( DELIMITER ) ); 
    ++anInt;
    return theSystemPath.substr( anInt, String::npos );
  }

  const String SystemPath::first() const
  {
    int anInt( theSystemPath.find( DELIMITER ) ); 
    String aString( "/" );

    if( anInt != 0 )
      {
	aString = theSystemPath.substr( 0, anInt );
      }

    return aString;
  }

  SystemPath SystemPath::next() const 
  {
    String::size_type aPosition = theSystemPath.find( DELIMITER );

    if( aPosition != String::npos )
      {
	++aPosition;
	return SystemPath( theSystemPath.substr( aPosition, 
						 String::npos ) ); 
      }

    return SystemPath( "" );
  }


  void SystemPath::standardize()
  {
    // FIXME: incomplete
  }


  ////////////////////////////////  FQID

  FQID::FQID( StringCref systemname, StringCref id )
    :
    SystemPath( systemname ), 
    theId( id )
  {
    ; // do nothing
  }

  FQID::FQID( StringCref fqid ) 
    : 
    SystemPath( SystemPathOf( fqid ) ),
    theId( IdOf( fqid ) )
  {
    standardize();
  }

  FQID::FQID( FQIDCref fqid ) 
    : 
    SystemPath( SystemPath( fqid ) ),
    theId( fqid.getIdString() )
  {
    standardize();
  }

  const String FQID::IdOf( StringCref fqid )
  {
    String::size_type aBorder = fqid.find( ':' );

    if( aBorder == String::npos )
      {
	throw BadID( __PRETTY_FUNCTION__,
		     "no \':\' found in FQID \"" + fqid + "\"." );
      }

    ++aBorder;
    if( fqid.find( ':', aBorder ) != String::npos )
      {
	throw BadID( __PRETTY_FUNCTION__,
		     "too many \':\'s in FQID \"" + fqid + "\"." );
      }

    return fqid.substr( aBorder, String::npos );
  }

  const String FQID::SystemPathOf( StringCref fqid )
  {
    String::size_type aBorder = fqid.find( ':' );

    if( aBorder == String::npos )
      {
	throw BadID( __PRETTY_FUNCTION__,
		     "no \':\' found in FQID \"" + fqid + "\"." );
      }

    if( fqid.find( ':', aBorder + 1 ) != String::npos )
      {
	throw BadID( __PRETTY_FUNCTION__,
		     "to many \':\'s in FQID \"" + fqid + "\"." );
      }

    return fqid.substr( 0, aBorder );
  }


  const String FQID::getFqidString() const
  {
    return ( SystemPath::getString() + ":" + getIdString() );
  }


  ////////////////////////////////  FQPI

  FQPI::FQPI( const PrimitiveType type, const FQID& fqid )
    :
    FQID( fqid ),
    thePrimitiveType( type )
  {
    ; // do nothing
  }

  //FIXME: ??
  FQPI::FQPI( StringCref fqpistring )
    : 
    FQID( fqidOf( fqpistring ) ),
    thePrimitiveType( PrimitiveTypeOf( PrimitiveTypeStringOf( fqpistring ) ) )
  {
    ; // do nothing
  }

  FQPI::FQPI( FQPICref fqpi )
    :
    FQID( static_cast<FQID>( fqpi ) ),
    thePrimitiveType( fqpi.getPrimitiveType() )
  {
    ; // do nothing
  }

  const String FQPI::fqidOf( StringCref fqpi )
  {
    String::size_type aBorder( fqpi.find( ':' ) );

    if( aBorder == String::npos )
      {
	throw BadID( __PRETTY_FUNCTION__,
		     "no \':\' found in FQPI \"" + fqpi + "\"." );
      }

    ++aBorder;
    if( fqpi.find( ':', aBorder ) == String::npos )
      {
	throw BadID( __PRETTY_FUNCTION__,
		     "not enough \':\'s found in FQPI \"" + fqpi + "\"." );
      }

    return fqpi.substr( aBorder, String::npos );
  }

  const String FQPI::getFqpiString() const 
  {
    String aString( PrimitiveTypeStringOf( thePrimitiveType ) + ':' 
		    + FQID::getString() );
    return aString;
  }


} // namespace libecs

#ifdef TEST_FQPI

using namespace libecs;

main()
{
  SystemPath aSystemPath( "/A/B" );
  cout << aSystemPath.getString() << endl;

  SystemPath aSystemPath2( "/A/../B" );
  cout << aSystemPath2.getString() << endl;

  FQID aFQID( "/A/B:S" );
  cout << aFQID.getString() << endl;
  cout << aFQID.getSystemPathString() << endl;
  cout << aFQID.getIdString() << endl;

  FQPI aFQPI( "Substance:/A/B:S" );
  cout << aFQPI.getString() << endl;
  cout << PrimitiveTypeStringOf( aFQPI.getPrimitiveType() ) << endl;
  cout << aFQPI.getSystemPathString() << endl;
  cout << aFQPI.getIdString() << endl;
  cout << aFQPI.getFqidString() << endl;
}


#endif


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
