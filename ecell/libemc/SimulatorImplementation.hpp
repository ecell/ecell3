#ifndef ___SIMULATOR_IMPLEMENTATION_H___
#define ___SIMULATOR_IMPLEMENTATION_H___

#include "libecs/libecs.hpp"
#include "util/Message.hpp"

class SimulatorImplementation
{

public:

  SimulatorImplementation();
  ~SimulatorImplementation() {};
  RootSystem* getRootSystemPtr() { return theRootSystem; }

  void makePrimitive( StringCref, FQPNCref, StringCref );
  void sendMessage( FQPNCref, MessageCref );
  Message getMessage( FQPNCref, StringCref );
  void step();

private:

  RootSystem* theRootSystem;

};   //end of class Simulator

#endif   /* ___SIMULATOR_IMPLEMENTATION_H___ */













