#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

#include "libemc/libemc.hpp"
#include "libemc/Simulator.hpp"

#include "PyEcs.hpp"

using namespace libemc;
using namespace libecs;


BOOST_PYTHON_MODULE( _ecs )
{

  // pyecs uses Numeric module
  import_array();


  // PySimulator class
  python::class_<Simulator>( "Simulator" )
    .def( python::init<>() )

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
    .def( "stop",                        &Simulator::stop )
    .def( "step",                        &Simulator::step )
    .def( "run", ( void ( Simulator::* )() )      &Simulator::run )
    .def( "run", ( void( Simulator::* )( Real ) ) &Simulator::run )
    
    .def( "setEventChecker",             &Simulator::setEventChecker )
    .def( "setEventHandler",             &Simulator::setEventHandler )
    // usually no need to call this explicitly
    .def( "initialize",                  &Simulator::initialize )
    
    ;  




  python::to_python_converter< Polymorph, Polymorph_to_python >();
  python::to_python_converter< DataPointVectorRCPtr, 
    DataPointVectorRCPtr_to_python >();

  register_Polymorph_from_python();
  register_EventCheckerRCPtr_from_python();
  register_EventHandlerRCPtr_from_python();

}

