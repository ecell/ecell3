#ifndef ___SIMULATOR_H___
#define ___SIMULATOR_H___

#include "libecs/libecs.hpp"

#include "LocalSimulatorImplementation.hpp"


class Simulator
{

  SimulatorImplementation* theSimulatorImplementation;

public:

  Simulator();
  ~Simulator() {}

  void makePrimitive( StringCref classname, 
		      FQPICref fqpn,
		      StringCref name )
  {
    theSimulatorImplementation->makePrimitive( classname, fqpn, name );
  }

  void sendMessage( FQPICref fqpn, 
		    MessageCref message )
  {
    theSimulatorImplementation->sendMessage( fqpn, message );
  }


  Message getMessage( FQPICref fqpn, 
		      StringCref propertyName )
  {
    theSimulatorImplementation->getMessage( fqpn, propertyName ); 
  }

  void step()
  {
    theSimulatorImplementation->step();
  }
};

#endif   /* ___SIMULATOR_H___ */





