//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 2002 Keio University
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
// written by Kouichi Takahashi <shafi@e-cell.org> for
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//

#include <signal.h>

#include "libemc/libemc.hpp"
#include "libemc/Simulator.hpp"
#include "libecs/Exceptions.hpp"

#include "PyEcs.hpp"

using namespace libemc;
using namespace libecs;



// exception translators

//void translateException( libecs::ExceptionCref anException )
//{
//  PyErr_SetString( PyExc_RuntimeError, anException.what() );
//}

void translateException( const std::exception& anException )
{
  PyErr_SetString( PyExc_RuntimeError, anException.what() );
}

static void PyEcsSignalHandler( int aSignal )
{
  static bool isCalled( false );
  if( isCalled )  // prevent looping
    {
      Py_FatalError( "PyECS: Fatal error.  Aborting uncleanly." );
    }
  isCalled = true;
  
  switch( aSignal )
    {
    case SIGSEGV:
      std::cerr << "PyECS: SIGSEGV. Invalid memory reference." << std::endl;
      break;
    case SIGFPE:
      std::cerr << "PyECS: SIGFPE. Floating point exception." << std::endl;
      break;
    case SIGINT:
      // exit without message
      break;
    default:
      std::cerr << "PyECS: Unexpected error: signal " << aSignal <<
	"." << std::endl;
      break;
    }
  
  Py_Exit( 1 );
}


BOOST_PYTHON_MODULE( _ecs )
{
  using namespace boost::python;

  // pyecs uses Numeric module
  import_array();

  signal( SIGSEGV, PyEcsSignalHandler );
  signal( SIGFPE,  PyEcsSignalHandler );
  signal( SIGINT,  PyEcsSignalHandler );


  def( "setDMSearchPath", &libemc::setDMSearchPath );
  def( "getDMSearchPath", &libemc::getDMSearchPath );


  // PySimulator class
  class_<Simulator>( "Simulator" )
    .def( init<>() )

    // Stepper-related methods
    .def( "createStepper",                &Simulator::createStepper )
    .def( "deleteStepper",                &Simulator::deleteStepper )
    .def( "getStepperList",               &Simulator::getStepperList )
    .def( "getStepperPropertyList",       &Simulator::getStepperPropertyList )
    .def( "getStepperPropertyAttributes", 
	  &Simulator::getStepperPropertyAttributes )
    .def( "setStepperProperty",           &Simulator::setStepperProperty )
    .def( "getStepperProperty",           &Simulator::getStepperProperty )
    .def( "getStepperClassName",          &Simulator::getStepperClassName )


    // Entity-related methods
    .def( "createEntity",                 &Simulator::createEntity )
    .def( "deleteEntity",                 &Simulator::deleteEntity )
    .def( "getEntityList",                &Simulator::getEntityList )
    .def( "isEntityExist",                &Simulator::isEntityExist )
    .def( "getEntityPropertyList",        &Simulator::getEntityPropertyList )
    .def( "setEntityProperty",            &Simulator::setEntityProperty )
    .def( "getEntityProperty",            &Simulator::getEntityProperty )
    .def( "getEntityPropertyAttributes", 
	  &Simulator::getEntityPropertyAttributes )
    .def( "getEntityClassName",           &Simulator::getEntityClassName )


    // Logger-related methods
    .def( "getLoggerList",                &Simulator::getLoggerList )  
    .def( "createLogger",                 &Simulator::createLogger )  
    .def( "getLoggerData", 
	  ( const DataPointVectorRCPtr ( Simulator::* )( StringCref ) const )
	  &Simulator::getLoggerData )
    .def( "getLoggerData", 
	  ( const DataPointVectorRCPtr 
	    ( Simulator::* )( StringCref, RealCref, RealCref ) const ) 
	  &Simulator::getLoggerData )
    .def( "getLoggerData",
	  ( const DataPointVectorRCPtr
	    ( Simulator::* )( StringCref, RealCref, 
			      RealCref, RealCref ) const ) 
	  &Simulator::getLoggerData )
    .def( "getLoggerStartTime",          &Simulator::getLoggerStartTime )  
    .def( "getLoggerEndTime",            &Simulator::getLoggerEndTime )    
    .def( "getLoggerMinimumInterval",    &Simulator::getLoggerMinimumInterval )
    .def( "setLoggerMinimumInterval",    &Simulator::setLoggerMinimumInterval )
    .def( "getLoggerSize",               &Simulator::getLoggerSize )


    // Simulation-related methods
    .def( "getCurrentTime",              &Simulator::getCurrentTime )
    .def( "getNextEvent",                &Simulator::getNextEvent )
    .def( "stop",                        &Simulator::stop )
    .def( "step", ( void ( Simulator::* )( void ) )       &Simulator::step )
    .def( "step", ( void ( Simulator::* )( const Int ) )  &Simulator::step )
    .def( "run",  ( void ( Simulator::* )() )             &Simulator::run )
    .def( "run",  ( void ( Simulator::* )( const Real ) ) &Simulator::run )
    
    
    .def( "setEventChecker",             &Simulator::setEventChecker )
    .def( "setEventHandler",             &Simulator::setEventHandler )
    // usually no need to call this explicitly
    .def( "initialize",                  &Simulator::initialize )
    
    ;  




  to_python_converter< Polymorph, Polymorph_to_python >();
  to_python_converter< DataPointVectorRCPtr, 
    DataPointVectorRCPtr_to_python >();

  register_Polymorph_from_python();
  register_EventCheckerRCPtr_from_python();
  register_EventHandlerRCPtr_from_python();

  register_exception_translator<Exception>     ( &translateException );
  register_exception_translator<std::exception>( &translateException );

}

