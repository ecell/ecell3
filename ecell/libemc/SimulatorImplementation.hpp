#ifndef ___SIMULATOR_IMPLEMENTATION_H___
#define ___SIMULATOR_IMPLEMENTATION_H___

#include "libecs/libecs.hpp"
#include "libecs/RootSystem.hpp"
#include "util/Message.hpp"

class SimulatorImplementation
{

public:

  SimulatorImplementation();
  ~SimulatorImplementation() {};
  RootSystem* getRootSystemPtr() { return theRootSystem; }

  void makePrimitive( StringCref classname, FQPNCref fqpn, StringCref name );
  void sendMessage( FQPNCref fqpn, MessageCref message );
  //  Message getMessage( FQPNCref fqpn, StringCref propertyName );
  void step();

private:

  RootSystem* theRootSystem;

};   //end of class Simulator

#endif   /* ___SIMULATOR_IMPLEMENTATION_H___ */













