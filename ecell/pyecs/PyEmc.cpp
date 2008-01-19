//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2008 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell System is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell System is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell System -- see the file COPYING.
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

// module initializer / finalizer
static struct _
{
  inline _()
  {
    if (!libecs::initialize())
      {
	throw std::runtime_error( "Failed to initialize libecs" );
      }
  }

  inline ~_()
  {
    libecs::finalize();
  }
} _;


BOOST_PYTHON_MODULE( _emc )
{
  using namespace boost::python;

  // this module uses Numeric module
  import_array();

  register_EventCheckerSharedPtr_from_python();
  register_EventHandlerSharedPtr_from_python();



  // Simulator class
  class_<libemc::Simulator>( "Simulator" )
    .def( init<>() )
    .def( "getClassInfo",
	  ( const libecs::PolymorphMap
	    ( libemc::Simulator::* )( libecs::StringCref, libecs::StringCref ) )
	  &libemc::Simulator::getClassInfo )
    .def( "getClassInfo",
	  ( const libecs::PolymorphMap
	    ( libemc::Simulator::* )( libecs::StringCref, libecs::StringCref,
                                  const libecs::Integer ) )
	  &libemc::Simulator::getClassInfo )
    // Stepper-related methods
    .def( "createStepper",
	  &libemc::Simulator::createStepper )
    .def( "deleteStepper",
	  &libemc::Simulator::deleteStepper )
    .def( "getStepperList",
	  &libemc::Simulator::getStepperList )
    .def( "getStepperPropertyList",
	  &libemc::Simulator::getStepperPropertyList )
    .def( "getStepperPropertyAttributes", 
	  &libemc::Simulator::getStepperPropertyAttributes )
    .def( "setStepperProperty",
	  &libemc::Simulator::setStepperProperty )
    .def( "getStepperProperty",
	  &libemc::Simulator::getStepperProperty )
    .def( "loadStepperProperty",
	  &libemc::Simulator::loadStepperProperty )
    .def( "saveStepperProperty",
	  &libemc::Simulator::saveStepperProperty )
    .def( "getStepperClassName",
	  &libemc::Simulator::getStepperClassName )

    // Entity-related methods
    .def( "createEntity",
	  &libemc::Simulator::createEntity )
    .def( "deleteEntity",
	  &libemc::Simulator::deleteEntity )
    .def( "getEntityList",
	  &libemc::Simulator::getEntityList )
    .def( "isEntityExist",
	  &libemc::Simulator::isEntityExist )
    .def( "getEntityPropertyList",
	  &libemc::Simulator::getEntityPropertyList )
    .def( "setEntityProperty",
	  &libemc::Simulator::setEntityProperty )
    .def( "getEntityProperty",
	  &libemc::Simulator::getEntityProperty )
    .def( "loadEntityProperty",
	  &libemc::Simulator::loadEntityProperty )
    .def( "saveEntityProperty",
	  &libemc::Simulator::saveEntityProperty )
    .def( "getEntityPropertyAttributes", 
	  &libemc::Simulator::getEntityPropertyAttributes )
    .def( "getEntityClassName",
	  &libemc::Simulator::getEntityClassName )

    // Logger-related methods
    .def( "getLoggerList",
	  &libemc::Simulator::getLoggerList )  
    .def( "createLogger",
	  ( void ( libemc::Simulator::* )( libecs::StringCref ) )
	  &libemc::Simulator::createLogger )  
    .def( "createLogger",		 
	  ( void ( libemc::Simulator::* )( libecs::StringCref,
					   libecs::Polymorph ) )
	  &libemc::Simulator::createLogger )  
    .def( "getLoggerData", 
	  ( const libecs::DataPointVectorSharedPtr
	    ( libemc::Simulator::* )( libecs::StringCref ) const )
	  &libemc::Simulator::getLoggerData )
    .def( "getLoggerData", 
	  ( const libecs::DataPointVectorSharedPtr 
	    ( libemc::Simulator::* )( libecs::StringCref,
				      libecs::RealCref,
				      libecs::RealCref ) const )
	  &libemc::Simulator::getLoggerData )
    .def( "getLoggerData",
	  ( const libecs::DataPointVectorSharedPtr
	    ( libemc::Simulator::* )( libecs::StringCref,
				      libecs::RealCref, 
				      libecs::RealCref,
				      libecs::RealCref ) const )
	  &libemc::Simulator::getLoggerData )
    .def( "getLoggerStartTime",
	  &libemc::Simulator::getLoggerStartTime )  
    .def( "getLoggerEndTime",
	  &libemc::Simulator::getLoggerEndTime )    
    .def( "getLoggerMinimumInterval",
          &libemc::Simulator::getLoggerMinimumInterval )
    .def( "setLoggerMinimumInterval",
          &libemc::Simulator::setLoggerMinimumInterval )
    .def( "getLoggerPolicy",
	  &libemc::Simulator::getLoggerPolicy )
    .def( "setLoggerPolicy",
	  &libemc::Simulator::setLoggerPolicy )
    .def( "getLoggerSize",
	  &libemc::Simulator::getLoggerSize )

    // Simulation-related methods
    .def( "getCurrentTime",
	  &libemc::Simulator::getCurrentTime )
    .def( "getNextEvent",
	  &libemc::Simulator::getNextEvent )
    .def( "stop",
	  &libemc::Simulator::stop )
    .def( "step",
	  ( void ( libemc::Simulator::* )( void ) )
	  &libemc::Simulator::step )
    .def( "step",
	  ( void ( libemc::Simulator::* )( const libecs::Integer ) )
	  &libemc::Simulator::step )
    .def( "run",
	  ( void ( libemc::Simulator::* )() )
      &libemc::Simulator::run )
    .def( "run",
	  ( void ( libemc::Simulator::* )( const libecs::Real ) ) 
	  &libemc::Simulator::run )
    .def( "setEventChecker",
	  &libemc::Simulator::setEventChecker )
    .def( "setEventHandler",
	  &libemc::Simulator::setEventHandler )
    ;  
}

