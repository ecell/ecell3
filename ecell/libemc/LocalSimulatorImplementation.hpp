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
		      FQPICref fqpn, 
		      StringCref name );

  void sendMessage( FQPICref fqpn, 
		    MessageCref message);

  Message getMessage( FQPICref fqpn, StringCref propertyName );

  void step();

private:

  RootSystemPtr       theRootSystem;

  SubstanceMakerPtr   theSubstanceMaker;
  ReactorMakerPtr     theReactorMaker;
  SystemMakerPtr      theSystemMaker;
  StepperMakerPtr     theStepper;
  AccumulatorMakerPtr theAccumulator;

};   //end of class LocalSimulatorImplementation

#endif   /* ___LOCAL_SIMULATOR_IMPLEMENTATION_H___ */













