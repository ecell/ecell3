#ifndef ___SIMULATOR_IMPLEMENTATION_H___
#define ___SIMULATOR_IMPLEMENTATION_H___

#include <string>
#include <stl.h>
#include <queue>

#include "Koyurugi/RootSystem.h"
#include "Command.h"

class SimulatorImplementation
{

public:

  SimulatorImplementation();
  ~SimulatorImplementation() {};

  void pushCommand( Command* command );
  void popCommand();

  RootSystem* getRootSystemPtr() { return theRootSystem; }
  
private:

  RootSystem* theRootSystem;

  //queue< Command* > theCommandQueue;
  queue< Command*, list< Command* > > theCommandQueue;

  //  void popCommand();
  //  void pushCommand( Command* command );

};   //end of class Simulator

#endif   /* ___SIMULATOR_IMPLEMENTATION_H___ */













