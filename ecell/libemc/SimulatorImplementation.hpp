#ifndef ___SIMULATOR_IMPLEMENTATION_H___
#define ___SIMULATOR_IMPLEMENTATION_H___

#include "libecs/libecs.hpp"
#include "libecs/RootSystem.hpp"
#include "util/Message.hpp"

/**
   Pure virtual base class (interface definition) of simulator
   implementation.
*/

class SimulatorImplementation
{

public:

  SimulatorImplementation() {}
  virtual ~SimulatorImplementation() {}

  virtual RootSystemPtr getRootSystemPtr() = 0;

  virtual void makePrimitive( StringCref classname, 
			      FQPICref fqpn, 
			      StringCref name ) = 0;

  virtual void sendMessage( FQPICref fqpn, 
			    MessageCref message ) = 0;

  virtual Message getMessage( FQPICref fqpn, 
			      StringCref propertyName ) = 0;
  virtual void step() = 0;

};   //end of class Simulator

#endif   /* ___SIMULATOR_IMPLEMENTATION_H___ */













