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
Object PySimulator::thePendingEventCheckerStore;
Object PySimulator::theEventHandlerStore;

PySimulator::PySimulator()
{
  ; // do nothing
}

void PySimulator::init_type()
{
  behaviors().name("Simulator");
  behaviors().doc("E-CELL Python class");

  add_varargs_method( "createStepper",         &PySimulator::createStepper );
  add_varargs_method( "createEntity",          &PySimulator::createEntity );
  add_varargs_method( "setProperty",           &PySimulator::setProperty );
  add_varargs_method( "getProperty",           &PySimulator::getProperty );
  add_varargs_method( "step",                  &PySimulator::step );
  add_varargs_method( "initialize",            &PySimulator::initialize );
  add_varargs_method( "getCurrentTime",        &PySimulator::getCurrentTime );
  add_varargs_method( "getLogger",             &PySimulator::getLogger );
  add_varargs_method( "getLoggerList",         &PySimulator::getLoggerList );
  add_varargs_method( "run",                   &PySimulator::run );
  add_varargs_method( "stop",                  &PySimulator::stop );
  add_varargs_method( "setPendingEventChecker",  
		      &PySimulator::setPendingEventChecker );
  add_varargs_method( "setEventHandler",       &PySimulator::setEventHandler );
}


Object PySimulator::step( const Py::Tuple& args )
{
  ECS_TRY;

  Simulator::step();
  return Py::Object();

  ECS_CATCH;
}

Object PySimulator::createStepper( const Py::Tuple& args )
{
  ECS_TRY;

  args.verify_length( 2, 3 );

  const String        aClassname ( static_cast<Py::String>( args[0] ) );
  const String        anID       ( static_cast<Py::String>( args[1] ) );


  UVariableVector aMessageBody;

  if( args.length() >= 3 )
    {
      const Py::Tuple aMessageSequence( static_cast<Py::Sequence>( args[2] ) );
      for( Py::Tuple::const_iterator i( aMessageSequence.begin() );
	   i != aMessageSequence.end() ; ++i )
	{
	  aMessageBody.push_back( PyUVariable( *i ) );
	}
    }

  Simulator::createStepper( aClassname, anID, aMessageBody );

  ECS_CATCH;
}

Object PySimulator::createEntity( const Py::Tuple& args )
{
  ECS_TRY;

  args.verify_length( 3 );

   const String        aClassname ( static_cast<Py::String>( args[0] ) );
   const String        aFullID    ( static_cast<Py::String>( args[1] ) );
   const String        aName      ( static_cast<Py::String>( args[2] ) );

   Simulator::createEntity( aClassname, aFullID, aName );

  return Py::Object();

  ECS_CATCH;
}
  
Object PySimulator::setProperty( const Py::Tuple& args )
{
  ECS_TRY;

  args.verify_length( 2 );

  const String        aFullID    ( static_cast<Py::String>( args[0] ) );
  const Py::Tuple aMessageSequence( static_cast<Py::Sequence>( args[1] ) );
  
  UVariableVector aMessageBody;
  for( Py::Tuple::const_iterator i( aMessageSequence.begin() );
       i != aMessageSequence.end() ; ++i )
    {
      aMessageBody.push_back( PyUVariable( *i ) );
    }

  Simulator::setProperty( aFullID, aMessageBody );

  return Py::Object();

  ECS_CATCH;
}

Object PySimulator::getProperty( const Py::Tuple& args )
{
  ECS_TRY;

  args.verify_length( 1 );
  
  const String         aFullID   ( static_cast<Py::String>( args[0] ) );

  UVariableVectorRCPtr aVectorPtr( Simulator::getProperty( aFullID ) );

  UVariableVector::size_type aSize( aVectorPtr->size() );

  Py::Tuple aTuple( aSize );

  for( UVariableVector::size_type i( 0 ) ; i < aSize ; ++i )
    {
      aTuple[i] = PyUVariable::toPyObject( (*aVectorPtr)[i] );
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


Object PySimulator::getCurrentTime( const Py::Tuple& )
{
  ECS_TRY;

  Py::Float aTime( Simulator::getCurrentTime() );

  return aTime;

  ECS_CATCH;
}


Object PySimulator::getLogger( const Py::Tuple& args )
{
  ECS_TRY;
  args.verify_length( 1 );

  const String        aFullID    ( static_cast<Py::String>( args[0] ) );

  LoggerPtr aLogger( Simulator::getLogger( aFullID ) );

  PyLogger* aPyLogger( new PyLogger( aLogger ) );

  return asObject( aPyLogger );

  ECS_CATCH;
}

Object PySimulator::getLoggerList( const Py::Tuple& args )
{
  ECS_TRY;
  args.verify_length( 0 );

  libecs::StringVectorRCPtr aLoggerListPtr( Simulator::getLoggerList() );

  Py::Tuple aTuple( aLoggerListPtr->size() );

  for( StringVector::size_type i( 0 ) ; i < aLoggerListPtr->size() ; ++i )
    {
      aTuple[i] = Py::String( (*aLoggerListPtr)[i] );
    }

  return aTuple;

  ECS_CATCH;
}

Object PySimulator::run( const Py::Tuple& args )
{
  ECS_TRY;
  args.verify_length( 0, 1 );

  if( args.length() != 0 )
    { 

      Py::Float aDuration = static_cast<Py::Float>( args[0] );
      
      Simulator::run( aDuration );
      
    } else {
      
      Simulator::run();

    }

  return Py::Object();

  ECS_CATCH;
}

Object PySimulator::stop( const Py::Tuple& args )
{
  ECS_TRY;
  args.verify_length( 0 );
 
  Simulator::stop();

  return Py::Object();

  ECS_CATCH;
}

Object PySimulator::setPendingEventChecker( const Py::Tuple& args )
{
  ECS_TRY;

  args.verify_length( 1 );

  thePendingEventCheckerStore =  static_cast<Py::Object>( args[0] );

  if( !thePendingEventCheckerStore.isCallable() )
    {
      throw Py::TypeError("Object is not Callable type.");
    }

  thePendingEventChecker = 
    static_cast<Py::Callable *>( &thePendingEventCheckerStore );

  Simulator::setPendingEventChecker
    ( static_cast<PendingEventCheckerFuncPtr>( callPendingEventChecker ) );
  
  return Py::Object();

  ECS_CATCH;
}

Object PySimulator::setEventHandler( const Py::Tuple& args )
{
  ECS_TRY;

  args.verify_length( 1 );

  theEventHandlerStore = static_cast<Py::Object>( args[0] );

  if( !theEventHandlerStore.isCallable() )
    {
      throw Py::TypeError("Object is not Callable type.");
    }

  theEventHandler = static_cast<Py::Callable *>( &theEventHandlerStore );


  Simulator::setEventHandler
    ( static_cast<EventHandlerFuncPtr>( callEventHandler ) );

  return Py::Object();

  ECS_CATCH;
}

bool PySimulator::callPendingEventChecker()
{
  Object anObject( thePendingEventChecker->apply( Py::Tuple() ) );

  return anObject.isTrue();
}

void PySimulator::callEventHandler()
{
  theEventHandler->apply( Py::Tuple() );
}

