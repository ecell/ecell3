#ifndef ___SIMULATOR_IMPLEMENTATION_H___
#define ___SIMULATOR_IMPLEMENTATION_H___

#include "libecs/libecs.hpp"
#include "libecs/RootSystem.hpp"
#include "util/Message.hpp"
#include "LocalSimulatorImplementation.hpp"

class SimulatorImplementation
  :
  public LocalSimulatorImplementation
{

public:

  SimulatorImplementation();
  ~SimulatorImplementation() {};
  RootSystem* getRootSystemPtr() { return theRootSystem; }

  virtual void makePrimitive( StringCref classname, FQPNCref fqpn, StringCref name ) = 0;
  virtual void sendMessage( FQPNCref fqpn, MessageCref message ) = 0;
  //  virtual Message getMessage( FQPNCref fqpn, StringCref propertyName ) = 0;
  virtual void step() = 0;

private:

  RootSystem* theRootSystem;
  SubstanceMaker* theSubstanceMaker;
  ReactorMaker* theReactorMaker;
  SystemMaker* SystemtheMaker;
  Stepper* theStepper;
  Accumulator* theAccumulator;

};   //end of class Simulator

#endif   /* ___SIMULATOR_IMPLEMENTATION_H___ */













