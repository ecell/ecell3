//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2002 Keio University
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
// written by Koichi Takahashi <shafi@e-cell.org> for
// E-Cell Project.
//

#include <iostream>
#include <signal.h>

#include "libemc/libemc.hpp"
#include "libemc/Simulator.hpp"
#include "libecs/Exceptions.hpp"

#include "libecs/Polymorph.hpp"
#include "libecs/Process.hpp"
#include "libecs/VariableReference.hpp"

#include "PyEmc.hpp"


using namespace libecs;
using namespace libemc;



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




BOOST_PYTHON_MODULE( _emc )
{
  using namespace boost::python;

  // this module uses Numeric module
  import_array();

  // don't catch SEGV and FPE, as it makes debugging harder.
  //  signal( SIGSEGV, PyEcsSignalHandler );
  //  signal( SIGFPE,  PyEcsSignalHandler );
  signal( SIGINT,  PyEcsSignalHandler );

  register_EventCheckerSharedPtr_from_python();
  register_EventHandlerSharedPtr_from_python();



  // Simulator class
  class_<Simulator>( "Simulator" )
    .def( init<>() )
    .def( "getClassInfo", 	( const PolymorphMap ( Simulator::* )
                            ( StringCref, StringCref ) )	  
                                          &Simulator::getClassInfo )
    .def( "getClassInfo", 	( const PolymorphMap ( Simulator::* )
                            ( StringCref, StringCref, const Integer ) )	  
                                          &Simulator::getClassInfo )
    // Stepper-related methods
    .def( "createStepper",                &Simulator::createStepper )
    .def( "deleteStepper",                &Simulator::deleteStepper )
    .def( "getStepperList",               &Simulator::getStepperList )
    .def( "getStepperPropertyList",       &Simulator::getStepperPropertyList )
    .def( "getStepperPropertyAttributes", 
	  &Simulator::getStepperPropertyAttributes )
    .def( "setStepperProperty",           &Simulator::setStepperProperty )
    .def( "getStepperProperty",           &Simulator::getStepperProperty )
    .def( "loadStepperProperty",          &Simulator::loadStepperProperty )
    .def( "saveStepperProperty",          &Simulator::saveStepperProperty )
    .def( "getStepperClassName",          &Simulator::getStepperClassName )


    // Entity-related methods
    .def( "createEntity",                 &Simulator::createEntity )
    .def( "deleteEntity",                 &Simulator::deleteEntity )
    .def( "getEntityList",                &Simulator::getEntityList )
    .def( "isEntityExist",                &Simulator::isEntityExist )
    .def( "getEntityPropertyList",        &Simulator::getEntityPropertyList )
    .def( "setEntityProperty",            &Simulator::setEntityProperty )
    .def( "getEntityProperty",            &Simulator::getEntityProperty )
    .def( "loadEntityProperty",           &Simulator::loadEntityProperty )
    .def( "saveEntityProperty",           &Simulator::saveEntityProperty )
    .def( "getEntityPropertyAttributes", 
	  &Simulator::getEntityPropertyAttributes )
    .def( "getEntityClassName",           &Simulator::getEntityClassName )


    // Logger-related methods
    .def( "getLoggerList",                &Simulator::getLoggerList )  
    .def( "createLogger",                 
	  ( void ( Simulator::* )( StringCref ) )
      &Simulator::createLogger )  
    .def( "createLogger",                 
	  ( void ( Simulator::* )( StringCref, Polymorph ) )
      &Simulator::createLogger )  
    .def( "getLoggerData", 
	  ( const DataPointVectorSharedPtr ( Simulator::* )( StringCref ) const )
	  &Simulator::getLoggerData )
    .def( "getLoggerData", 
	  ( const DataPointVectorSharedPtr 
	    ( Simulator::* )( StringCref, RealCref, RealCref ) const ) 
	  &Simulator::getLoggerData )
    .def( "getLoggerData",
	  ( const DataPointVectorSharedPtr
	    ( Simulator::* )( StringCref, RealCref, 
			      RealCref, RealCref ) const ) 
	  &Simulator::getLoggerData )
    .def( "getLoggerStartTime",          &Simulator::getLoggerStartTime )  
    .def( "getLoggerEndTime",            &Simulator::getLoggerEndTime )    
    .def( "getLoggerMinimumInterval",    &Simulator::getLoggerMinimumInterval )
    .def( "setLoggerMinimumInterval",    &Simulator::setLoggerMinimumInterval )
    .def( "getLoggerPolicy",    		&Simulator::getLoggerPolicy )
    .def( "setLoggerPolicy",   	&Simulator::setLoggerPolicy )
    .def( "getLoggerSize",               &Simulator::getLoggerSize )


    // Simulation-related methods
    .def( "getCurrentTime",              &Simulator::getCurrentTime )
    .def( "getNextEvent",                &Simulator::getNextEvent )
    .def( "stop",                        &Simulator::stop )
    .def( "step", ( void ( Simulator::* )( void ) )          &Simulator::step )
    .def( "step", ( void ( Simulator::* )( const Integer ) ) &Simulator::step )
    .def( "run",  ( void ( Simulator::* )() )                &Simulator::run )
    .def( "run",  ( void ( Simulator::* )( const Real ) )    &Simulator::run )
    
    
    .def( "setEventChecker",             &Simulator::setEventChecker )
    .def( "setEventHandler",             &Simulator::setEventHandler )

    //    .def( "getLoadedDMList",             &Simulator::getLoadedDMList )
    ;  


}

