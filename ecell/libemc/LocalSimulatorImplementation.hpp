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
  virtual ~LocalSimulatorImplementation() {}

  RootSystemRef getRootSystem() { return theRootSystem; }

  void makePrimitive( StringCref classname, 
		      FQPICref fqpi, 
		      StringCref name );

  void sendMessage( FQPICref fqpi, 
		    MessageCref message);

  Message getMessage( FQPICref fqpi, StringCref propertyName );

  void step();

private:

  RootSystem     theRootSystem;

};   //end of class LocalSimulatorImplementation

#endif   /* ___LOCAL_SIMULATOR_IMPLEMENTATION_H___ */
