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


#include "libecs/libecs.hpp"
#include "libecs/FullID.hpp"
#include "libecs/Message.hpp"

#include "PyUVariable.hpp"
#include "PyEcs.hpp"
#include "PyLogger.hpp"

#include "PySimulator.hpp"

using namespace libemc;
using namespace libecs;

Callable* PySimulator::thePendingEventChecker;
Callable* PySimulator::theEventHandler;

PySimulator::PySimulator()
{
  ; // do nothing
}

void PySimulator::init_type()
{
  behaviors().name("Simulator");
  behaviors().doc("E-CELL Python class");

  add_varargs_method( "createEntity",          &PySimulator::createEntity );
  add_varargs_method( "setProperty",           &PySimulator::setProperty );
  add_varargs_method( "getProperty",           &PySimulator::getProperty );
  add_varargs_method( "step",                  &PySimulator::step );
  add_varargs_method( "initialize",            &PySimulator::initialize );
  add_varargs_method( "getLogger",             &PySimulator::getLogger );
  add_varargs_method( "run",                   &PySimulator::run );
  add_varargs_method( "stop",                  &PySimulator::stop );
  add_varargs_method( "setPendingEventChecker",  &PySimulator::setPendingEventChecker );
  add_varargs_method( "setEventHandler",       &PySimulator::setEventHandler );
}

Object PySimulator::step( const Py::Tuple& args )
{
  ECS_TRY;

  Simulator::step();
  return Py::Object();

  ECS_CATCH;
}

Object PySimulator::createEntity( const Py::Tuple& args )
{
  ECS_TRY;

  args.verify_length( 3 );

  const Py::Sequence& aFullID( args[1] );
  aFullID.verify_length( 3 );

  const String        aClassname ( static_cast<Py::String>( args[0] ) );
  const PrimitiveType aType      ( static_cast<Py::Int>   ( aFullID[0] ) );
  const String        aPath      ( static_cast<Py::String>( aFullID[1] ) );
  const String        anID       ( static_cast<Py::String>( aFullID[2] ) );
  const String        aName      ( static_cast<Py::String>( args[2] ) );

  Simulator::createEntity( aClassname, aType, aPath, anID, aName );

  return Py::Object();

  ECS_CATCH;
}
  
Object PySimulator::setProperty( const Py::Tuple& args )
{
  ECS_TRY;

  args.verify_length( 2 );

  const Py::Sequence& aFullPN( args[0] );
  aFullPN.verify_length( 4 );

  const PrimitiveType aType ( static_cast<Py::Int>( aFullPN[0] ) );
  const String aPath        ( static_cast<Py::String>( aFullPN[1] ) );
  const String anID         ( static_cast<Py::String>( aFullPN[2] ) );
  const String aPropertyName( static_cast<Py::String>( aFullPN[3] ) );
  const Py::Tuple aMessageSequence( static_cast<Py::Sequence>( args[1] ) );
  
  UConstantVector aMessageBody;
  for( Py::Tuple::const_iterator i( aMessageSequence.begin() );
       i != aMessageSequence.end() ; ++i )
    {
      aMessageBody.push_back( PyUConstant( *i ) );
    }

  Simulator::setProperty( aType, aPath, anID, aPropertyName, aMessageBody );

  return Py::Object();

  ECS_CATCH;
}

Object PySimulator::getProperty( const Py::Tuple& args )
{
  ECS_TRY;

  args.verify_length( 1 );
  
  const Py::Sequence& aFullPN( args[0] );
  aFullPN.verify_length( 4 );

  const PrimitiveType aType    ( static_cast<Py::Int>( aFullPN[0] ) );
  const String aPath        ( static_cast<Py::String>( aFullPN[1] ) );
  const String anID         ( static_cast<Py::String>( aFullPN[2] ) );
  const String aPropertyName( static_cast<Py::String>( aFullPN[3] ) );

  UConstantVectorRCPtr aVectorPtr( Simulator::getProperty( aType,
							   aPath,
							   anID,
							   aPropertyName ) );

  UConstantVector::size_type aSize( aVectorPtr->size() );

  Py::Tuple aTuple( aSize );

  for( UConstantVector::size_type i( 0 ) ; i < aSize ; ++i )
    {
      aTuple[i] = PyUConstant::toPyObject( (*aVectorPtr)[i] );
    }

  return aTuple;

  ECS_CATCH;
}


Object PySimulator::initialize( const Py::Tuple& )
{
  ECS_TRY;

  Simulator::initialize();

  return Object();

  ECS_CATCH;
}



Object PySimulator::getLogger( const Py::Tuple& args )
{
  ECS_TRY;
  args.verify_length( 1 );

  const Py::Sequence& aFullPN( args[0] );
  aFullPN.verify_length( 4 );

  const PrimitiveType aType    ( static_cast<Py::Int>( aFullPN[0] ) );
  const String aPath        ( static_cast<Py::String>( aFullPN[1] ) );
  const String anID         ( static_cast<Py::String>( aFullPN[2] ) );
  const String aPropertyName( static_cast<Py::String>( aFullPN[3] ) );

  const Logger* aLogger( Simulator::getLogger( aType,
					       aPath,
					       anID,
					       aPropertyName ) );

  PyLogger* aPyLogger = new PyLogger( aLogger );

  return asObject( aPyLogger );

  ECS_CATCH;
}

Object PySimulator::run( const Py::Tuple& args )
{
  ECS_TRY;

  Simulator::run();

  return Py::Object();

  ECS_CATCH;
}

Object PySimulator::stop( const Py::Tuple& args )
{
  ECS_TRY;
 
  Simulator::stop();

  return Py::Object();

  ECS_CATCH;
}

Object PySimulator::setPendingEventChecker( const Py::Tuple& args )
{
  ECS_TRY;

  args.verify_length( 1 );

  theTmpPendingEventChecker = static_cast<Py::Object>( args[0] );
  thePendingEventChecker = static_cast<Py::Callable *>( &theTmpPendingEventChecker );
  // *thePendingEventChecker = theTmpPendingEventChecker;

  if( !thePendingEventChecker->isCallable() )
    throw Py::TypeError("Object is not Callable type.");
  
  Simulator::setPendingEventChecker( (bool (*)())callPendingEventChecker );
  
  return Py::Object();

  ECS_CATCH;
}

Object PySimulator::setEventHandler( const Py::Tuple& args )
{
  ECS_TRY;

  args.verify_length( 1 );

  theTmpEventHandler = static_cast<Py::Object>( args[0] );
  theEventHandler = static_cast<Py::Callable *>( &theTmpEventHandler );
  //  *theEventHandler = theTmpEventHandler;

  if( !theEventHandler->isCallable() )
    throw Py::TypeError("Object is not Callable type.");

  Simulator::setEventHandler( (void (*)())callEventHandler );

  return Py::Object();

  ECS_CATCH;
}

void PySimulator::callPendingEventChecker()
{
  ECS_TRY;

  thePendingEventChecker->apply();

  ECS_CATCH; 
}

void PySimulator::callEventHandler()
{
  ECS_TRY;

  theEventHandler->apply();

  ECS_CATCH; 
}




