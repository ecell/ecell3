#ifndef ___SIMULATOR_H___
#define ___SIMULATOR_H___

#include "SimulatorImplementation.hpp"

class Simulator
{

  SimulatorImplementation* theSimulatorImplementation;

public:

  Simulator();
  ~Simulator(){};

  void makePrimitive( StringCref, FQPNCref, StringCref );
  void sendMessage( FQPNCref, Message );
  Message getMessage( FQPNCref, StringCref );
  void step();
};

#endif   /* ___SIMULATOR_H___ */
