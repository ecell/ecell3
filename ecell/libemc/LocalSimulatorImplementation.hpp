#ifndef ___LOCAL_SIMULATOR_IMPLEMENTATION_H___
#define ___LOCAL_SIMULATOR_IMPLEMENTATION_H___

#include "libecs/libecs.hpp"

#include "SimulatorImplementation.hpp"

class LocalSimulatorImplementation
  :
  public SimulatorImplementation
{

public:

  LocalSimulatorImplementation();
  ~LocalSimulatorImplementation() {}

  RootSystemPtr getRootSystemPtr() { return theRootSystem; }

  void makePrimitive( StringCref classname, 
		      FQPNCref fqpn, 
		      StringCref name );

  void sendMessage( FQPNCref fqpn, 
		    MessageCref message);

  Message getMessage( FQPNCref fqpn, StringCref propertyName );

  void step();

private:

  RootSystem* theRootSystem;
  SubstanceMaker* theSubstanceMaker;
  ReactorMaker* theReactorMaker;
  SystemMaker* SystemtheMaker;
  Stepper* theStepper;
  Accumulator* theAccumulator;

};   //end of class LocalSimulatorImplementation

#endif   /* ___LOCAL_SIMULATOR_IMPLEMENTATION_H___ */













