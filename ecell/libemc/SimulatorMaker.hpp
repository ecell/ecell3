#ifndef ___SIMULATORMAKER_H___
#define ___SIMULATORMAKER_H___
#include <stl.h>
#include "Simulator.h"
#include "SharedModuleMaker.h"

class SimulatorMaker : public SharedModuleMaker<Simulator>
{
private:

protected:

public:

  SimulatorMaker();
  Simulator* make(const string& classname) throw(CantInstantiate);
  void install(const string& systementry);

  virtual const char* const className() const {return "SimulatorMaker";}
};

#define NewSimulatorModule(CLASS) NewDynamicModule(Simulator,CLASS)

#endif /* ___SIMULATORMAKER_H___ */

