#ifndef ___SIMULATOR_H___
#define ___SIMULATOR_H___

#include "LocalSimulatorImplementation.hpp"

class Simulator
{

  LocalSimulatorImplementation* theSimulatorImplementation;

public:

  Simulator();
  ~Simulator(){};

  void makePrimitive( StringCref classname, FQPNCref fqpn, StringCref name );
  void sendMessage( FQPNCref fqpn, Message message );
  //  Message getMessage( FQPNCref fqpn, StringCref propertyName );
  void step();
};

#endif   /* ___SIMULATOR_H___ */





