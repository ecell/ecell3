#ifndef ___SIMULATOR_H___
#define ___SIMULATOR_H___

#include "SimulatorImplementation.h"

#include <string>

class Simulator
{

  SimulatorImplementation* theSimulatorImplementation;

public:

  Simulator();
  ~Simulator(){};

  void pushCommand( Command* );
  void popCommand();
  void test();
};

#endif   /* ___SIMULATOR_H___ */










