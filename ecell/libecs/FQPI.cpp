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
#include "FQPN.hpp"


///////////////////////  SystemPath

SystemPath::SystemPath( StringCref rqsn ) 
  :
  theSystemPath( rqsn )
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
 int    anInt( theSystemPath.find( DELIMITER ) ); 

 if( anInt != 0 )
   {
     return theSystemPath.substr( 0, anInt );
   }

 return "/";
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


////////////////////////////////  FQIN

FQIN::FQIN( StringCref systemname, StringCref id )
  :
  SystemPath( systemname ), 
  theId( id )
{
  ; // do nothing
}

FQIN::FQIN( StringCref fqin ) 
  : 
  SystemPath( SystemPathOf( fqin ) ),
  theId( IdOf( fqin ) )
{
  standardize();
}

const String FQIN::IdOf( StringCref fqin )
{
  String::size_type aBorder = fqin.find( ':' );

  if( aBorder == String::npos )
    {
      throw BadFQIN(__PRETTY_FUNCTION__,
		    "no \':\' found in \"" + fqin + "\".");
    }

  if( fqin.find( ':', aBorder + 1 ) != String::npos )
    {
      throw BadFQIN(__PRETTY_FUNCTION__,
		    "too many \':\'s in \"" + fqin + "\".");
    }

  return fqin.substr( aBorder + 1, String::npos );
}

const String FQIN::SystemPathOf( StringCref fqin )
{
  String::size_type aBorder = fqin.find( ':' );

  if( aBorder == String::npos )
    {
      throw BadFQIN( __PRETTY_FUNCTION__,
		     "no \':\' found in \"" + fqin + "\"." );
    }

  if( fqin.find( ':', aBorder + 1 ) != String::npos )
    {
      throw BadFQIN(__PRETTY_FUNCTION__,
		    "to many \':\'s in \"" + fqin + "\".");
    }

  return fqin.substr( 0, aBorder );
}


const String FQIN::getFqin() const
{
  return ( SystemPath::getString() + ":" + getId() );
}


////////////////////////////////  FQPN

FQPN::FQPN( const Primitive::Type type, const FQIN& fqin )
  :
  FQIN( fqin ),
  theType( type )
{
  ; // do nothing
}

FQPN::FQPN( StringCref fqpn )
  : FQIN( fqinOf( fqpn ) ),
  theType( typeOf( fqpn ) )
{
  ; // do nothing
}

const String FQPN::fqinOf( StringCref fqpn )
{
  String::size_type aBorder( fqpn.find(':') );

  if( aBorder == String::npos )
    {
      throw BadFQPN(__PRETTY_FUNCTION__,
		    "no \':\' found in \"" + fqpn + "\".");
    }
  if( fqpn.find( ':', aBorder + 1 ) == String::npos )
    {
      throw BadFQPN(__PRETTY_FUNCTION__,
		    "no enough \':\'s found in \"" + fqpn + "\".");
    }

  return fqpn.substr( aBorder + 1, String::npos );
}

Primitive::Type FQPN::typeOf( StringCref fqpn )
{
  String::size_type aBorder( fqpn.find(':') );

  if( aBorder == String::npos )
    {
      throw BadFQPN(__PRETTY_FUNCTION__,
		    "no \':\' found in \"" + fqpn + "\".");
    }
  if( fqpn.find( ':', aBorder + 1 ) == String::npos )
    {
      throw BadFQPN(__PRETTY_FUNCTION__,
		    "no enough \':\'s found in \"" + fqpn + "\".");
    }
  
  String aTypeString = fqpn.substr( 0, aBorder );

  return Primitive::PrimitiveType( aTypeString );
}

const String FQPN::getFqpn() const 
{
  return Primitive::PrimitiveTypeString( theType ) 
    + ':' + FQIN::getString();
}


#ifdef TEST_FQPN

main()
{
  SystemPath aSystemPath( "/A/B" );
  cout << aSystemPath.getString() << endl;

  SystemPath aSystemPath2( "/A/../B" );
  cout << aSystemPath2.getString() << endl;

  FQIN aFQIN( "/A/B:S" );
  cout << aFQIN.getString() << endl;
  cout << aFQIN.getSystemPath() << endl;
  cout << aFQIN.getId() << endl;

  FQPN aFQPN( "Substance:/A/B:S" );
  cout << aFQPN.getString() << endl;
  cout << Primitive::PrimitiveTypeString( aFQPN.getType() ) << endl;
  cout << aFQPN.getSystemPath() << endl;
  cout << aFQPN.getId() << endl;
  cout << aFQPN.getFqin() << endl;
}


#endif


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
