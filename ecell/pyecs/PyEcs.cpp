#include <boost/python/class_builder.hpp>
#include <boost/python/cross_module.hpp>

#include "libemc/libemc.hpp"
#include "libemc/Simulator.hpp"

#include "PyEcs.hpp"

BOOST_PYTHON_BEGIN_CONVERSION_NAMESPACE

const libecs::PolymorphVector ref_to_PolymorphVector( const ref& aRef )
{
  tuple aPyTuple( aRef );

  std::size_t aSize( aPyTuple.size() );

  libecs::PolymorphVector aVector;
  aVector.reserve( aSize );
      
  for ( std::size_t i( 0 ); i < aSize; ++i )
    {
      ref anItemRef( aPyTuple[i] );
      PyObject* aPyObjectPtr( anItemRef.get() ); 
      
      aVector.push_back( from_python( aPyObjectPtr, 
				      type<libecs::Polymorph>() ) );
    }

  return aVector;
}

static PyObject* 
PolymorphVector_to_python( libecs::PolymorphVectorCref aVector )
{
  libecs::PolymorphVector::size_type aSize( aVector.size() );
  
  tuple aPyTuple( aSize );

  for( size_t i( 0 ) ; i < aSize ; ++i )
    {
      aPyTuple.set_item( i, BOOST_PYTHON_CONVERSION::to_python( aVector[i] ) );
    }

  return to_python( aPyTuple.get() );
}


BOOST_PYTHON_END_CONVERSION_NAMESPACE


BOOST_PYTHON_MODULE_INIT(_ecs)
{

  // pyecs uses Numeric module
  import_array();


  python::module_builder ecs( "_ecs" );

  // PySimulator class
  python::class_builder<libemc::Simulator> aSimulatorClass( ecs, "Simulator" );
  // PyLogger class
  python::class_builder<libemc::EmcLogger> aLoggerClass( ecs, "Logger" );

  //
  // PySimulator Definitions
  //

  aSimulatorClass.def( python::constructor<>() );

  // Entity-related methods
  aSimulatorClass.def( &libemc::Simulator::createEntity,   "createEntity" );
  aSimulatorClass.def( &libemc::Simulator::isEntityExist,  "isEntityExist" );
  aSimulatorClass.def( &libemc::Simulator::setEntityProperty,
		       "setEntityProperty" );
  aSimulatorClass.def( &libemc::Simulator::getEntityProperty, 
		       "getEntityProperty" );

  // Stepper-related methods
  aSimulatorClass.def( &libemc::Simulator::createStepper,   "createStepper" );
  aSimulatorClass.def( &libemc::Simulator::getStepperList,  "getStepperList" );
  aSimulatorClass.def( &libemc::Simulator::setStepperProperty, 
		       "setStepperProperty" );
  aSimulatorClass.def( &libemc::Simulator::getStepperProperty,
		       "getStepperProperty" );


  // Logger-related methods
  aSimulatorClass.def( &libemc::Simulator::getLoggerList,  "getLoggerList" );  
  aSimulatorClass.def( &libemc::Simulator::createLogger,  "createLogger" );  
  aSimulatorClass.def( ( const libecs::DataPointVectorRCPtr 
			 ( libemc::Simulator::* )( libecs::StringCref ) const )
		       &libemc::Simulator::getLoggerData,
		       "getLoggerData" );
  aSimulatorClass.def( ( const libecs::DataPointVectorRCPtr 
			 ( libemc::Simulator::* )( libecs::StringCref, 
						   libecs::RealCref, 
						   libecs::RealCref ) const )
		       &libemc::Simulator::getLoggerData,
		       "getLoggerData" );
  aSimulatorClass.def( ( const libecs::DataPointVectorRCPtr
			 ( libemc::Simulator::* )( libecs::StringCref, 
						   libecs::RealCref, 
						   libecs::RealCref, 
						   libecs::RealCref ) const )
		       &libemc::Simulator::getLoggerData, 
		       "getLoggerData" );
  aSimulatorClass.def( &libemc::Simulator::getLoggerStartTime,
		       "getLoggerStartTime" );  
  aSimulatorClass.def( &libemc::Simulator::getLoggerEndTime, 
		       "getLoggerEndTime" );    
  aSimulatorClass.def( &libemc::Simulator::getLoggerMinimumInterval, 
		       "getLoggerMinimumInterval" );    
  aSimulatorClass.def( &libemc::Simulator::setLoggerMinimumInterval, 
		       "setLoggerMinimumInterval" );    
  aSimulatorClass.def( &libemc::Simulator::getLoggerSize, "getLoggerSize" );



  // Simulation-related methods
  aSimulatorClass.def( &libemc::Simulator::getCurrentTime, "getCurrentTime" );
  aSimulatorClass.def( &libemc::Simulator::stop,           "stop" );
  aSimulatorClass.def( &libemc::Simulator::step,           "step" );
  aSimulatorClass.def( ( void ( libemc::Simulator::* )() )
		       &libemc::Simulator::run,            "run" );
  aSimulatorClass.def( ( void( libemc::Simulator::* )( libecs::Real ) )
		       &libemc::Simulator::run,            "run" );

  aSimulatorClass.def( &libemc::Simulator::setPendingEventChecker,
		       "setPendingEventChecker" );
  aSimulatorClass.def( &libemc::Simulator::setEventHandler, 
		       "setEventHandler" );

  // usually no need to call this explicitly
  aSimulatorClass.def( &libemc::Simulator::initialize,     "initialize" );  


}

