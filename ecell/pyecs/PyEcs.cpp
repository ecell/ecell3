#include <boost/python/class_builder.hpp>
#include <boost/python/cross_module.hpp>

#include "libemc/libemc.hpp"
#include "libemc/Simulator.hpp"
#include "libemc/EmcLogger.hpp"

//#include "PySimulator.hpp"
//#include "PyLogger.hpp"

#include "PyEcs.hpp"




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
  aSimulatorClass.def( &libemc::Simulator::createEntity,   "createEntity" );
  aSimulatorClass.def( &libemc::Simulator::setProperty,    "setProperty" );
  aSimulatorClass.def( &libemc::Simulator::getProperty,    "getProperty" );
  aSimulatorClass.def( &libemc::Simulator::createStepper,  "createStepper" );
  aSimulatorClass.def( &libemc::Simulator::getCurrentTime, "getCurrentTime" );
  aSimulatorClass.def( &libemc::Simulator::getLogger,      "getLogger" );
  aSimulatorClass.def( &libemc::Simulator::getLoggerList,  "getLoggerList" );  
  aSimulatorClass.def( &libemc::Simulator::stop,           "stop" );
  aSimulatorClass.def( &libemc::Simulator::step,           "step" );
  aSimulatorClass.def( &libemc::Simulator::initialize,     "initialize" );  
  aSimulatorClass.def( ( void ( libemc::Simulator::* )() )
		       &libemc::Simulator::run,            "run" );
  aSimulatorClass.def( ( void( libemc::Simulator::* )( libecs::Real ) )
		       &libemc::Simulator::run,            "run" );
  aSimulatorClass.def( &libemc::Simulator::setPendingEventChecker,
		       "setPendingEventChecker" );
  aSimulatorClass.def( &libemc::Simulator::setEventHandler, 
		       "setEventHandler" );


  //
  // PyLogger definitions
  //

  //  no constructor
  //  aPyLoggerClass.def( python::constructor<libecs::LoggerPtr>() );

  aLoggerClass.def( ( const libecs::DataPointVectorRCPtr 
		      ( libemc::EmcLogger::* )() )
		    &libemc::EmcLogger::getData,
		    "getData" );
  aLoggerClass.def( ( const libecs::DataPointVectorRCPtr 
		      ( libemc::EmcLogger::* )( libecs::RealCref, 
						libecs::RealCref ) )
		    &libemc::EmcLogger::getData,
		    "getData" );
  aLoggerClass.def( ( const libecs::DataPointVectorRCPtr
		      ( libemc::EmcLogger::* )( libecs::RealCref, 
						libecs::RealCref, 
						libecs::RealCref ) )
		    &libemc::EmcLogger::getData, 
		    "getData" );
  aLoggerClass.def( &libemc::EmcLogger::getStartTime, "getStartTime" );  
  aLoggerClass.def( &libemc::EmcLogger::getEndTime,   "getEndTime" );    
  aLoggerClass.def( &libemc::EmcLogger::getMinimumInterval,
		    "getMinimumInterval" );    
  aLoggerClass.def( &libemc::EmcLogger::getSize, "getSize" );

}

