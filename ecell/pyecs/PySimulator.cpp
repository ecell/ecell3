//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2001 Keio University
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
// E-CELL Project, Institute for Advanced Biosciences, Keio University.
//


#include <string>

#include "libecs.hpp"
#include "FQPI.hpp"
#include "Message.hpp"

#include "PySimulator.hpp"

#define ECS_TRY try {

#define ECS_CATCH\
    }\
  catch( ::ExceptionCref e )\
    {\
      throw Py::Exception( e.message() );\
    }\
  catch( const ::exception& e)\
    {\
      throw Py::SystemError( std::string("E-CELL internal error: ")\
			     + e.what() );\
    }\
  catch( ... ) \
    {\
      throw Py::SystemError( "E-CELL internal error." );\
    }

PySimulator::PySimulator()
{
  ; // do nothing
}

void PySimulator::init_type()
{
  behaviors().name("Simulator");
  behaviors().doc("E-CELL Python class");

  add_varargs_method( "makePrimitive", &PySimulator::makePrimitive );
  add_varargs_method( "sendMessage",   &PySimulator::sendMessage );
  add_varargs_method( "getMessage",    &PySimulator::getMessage );
  add_varargs_method( "step",          &PySimulator::step );
  add_varargs_method( "initialize",    &PySimulator::initialize );
}

Object PySimulator::step( const Tuple& args )
{
  ECS_TRY;

  Simulator::step();
  return Object();

  ECS_CATCH;
}

Object PySimulator::makePrimitive( const Tuple& args )
{
  ECS_TRY;

  args.verify_length( 3 );
  const string aClassname( static_cast<Py::String>( args[0] ) );
  const FQPI aFqpi( static_cast<Py::String>( args[1] ) );
  const string aName( static_cast<Py::String>( args[2] ) );

  Simulator::makePrimitive( aClassname, aFqpi, aName );

  return Py::Object();

  ECS_CATCH;
}
  
Object PySimulator::sendMessage( const Tuple& args )
{
  ECS_TRY;

  args.verify_length( 3 );
  const string aFqpi( static_cast<Py::String>( args[0] ) );
  const string aMessageKeyword( static_cast<Py::String>( args[1] ) );

  const Tuple aMessageSequence( static_cast<Py::Sequence>( args[2] ) );
  
  UniversalVariableVector aMessageBody;
  for( Py::Tuple::const_iterator i = aMessageSequence.begin() ;
       i != aMessageSequence.end() ; ++i )
    {
      aMessageBody.push_back( UniversalVariable( (*i).as_string() ) );
    }

  const Message aMessage( aMessageKeyword, aMessageBody );

  Simulator::sendMessage( FQPI( aFqpi ), aMessage );

  return Object();

  ECS_CATCH;
}

Object PySimulator::getMessage( const Tuple& args )
{
  ECS_TRY;

  args.verify_length( 2 );
  
  const FQPI aFqpi( static_cast<Py::String>( args[0] ) );
  const string aPropertyName( static_cast<Py::String>( args[1] ) );

  Message aMessage( Simulator::getMessage( aFqpi, aPropertyName ) );
  int aMessageSize = aMessage.getBody().size();

  Tuple aTuple( aMessageSize );

  for( int i = 0 ; i < aMessageSize ; ++i )
    {
      UniversalVariableCref aUniversalVariable( aMessage.getBody()[i] );
      Py::Object anObject;
      if( aUniversalVariable.isReal() )
	{
	  anObject = Py::Float( aUniversalVariable.asReal() );
	}
      else if( aUniversalVariable.isInt() )
	{
	  anObject = Py::Int( aUniversalVariable.asInt() );
	}
      else if( aUniversalVariable.isString() )
	{
	  anObject = Py::String( aUniversalVariable.asString() );
	}
      else
	{
	  assert( 0 );
	  ; //FIXME: assert: NEVER_GET_HERE
	}

      aTuple[i] = anObject;
    }

  return aTuple;

  ECS_CATCH;
}


Object PySimulator::initialize( const Tuple& )
{
  ECS_TRY;

  Simulator::initialize();

  return Object();

  ECS_CATCH;
}



