#include <stdlib.h>
#include "SimulatorMaker.h"

////////////////////// SimulatorMaker
 

SimulatorMaker::SimulatorMaker()
{
}

Simulator* SimulatorMaker::make(const string& classname) throw(CantInstantiate)
{
  assert(allocator(classname));
  Simulator* instance = allocator(classname)();

  if(instance == NULL)
    throw CantInstantiate();

  return instance;
}




