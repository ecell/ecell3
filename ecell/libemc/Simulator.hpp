#ifndef ___SIMULATOR_H___
#define ___SIMULATOR_H___

#include "SimulatorImplementation.hpp"

class Simulator
{

  SimulatorImplementation* theSimulatorImplementation;

public:

  Simulator();
  ~Simulator(){};

  //  void makePrimitive( StringCref classname, FQPNCref fqpn, StringCref name );
  void makePrimitive();
  //  void sendMessage( FQPNCref fqpn, Message message );
  void sendMessage();
  //  Message getMessage( FQPNCref fqpn, StringCref propertyName );
  void getMessage();
  void step();
};

#endif   /* ___SIMULATOR_H___ */





