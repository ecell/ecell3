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
		      FQPNCref fqpn,
		      StringCref name )
  {
    theSimulatorImplementation->makePrimitive( classname, fqpn, name );
  }

  void sendMessage( FQPNCref fqpn, 
		    MessageCref message )
  {
    theSimulatorImplementation->sendMessage( fqpn, message );
  }


  Message getMessage( FQPNCref fqpn, 
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





