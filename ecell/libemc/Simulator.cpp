#include "LocalSimulatorImplementation.hpp"
#include "Simulator.hpp"

Simulator::Simulator()
{
  //FIXME: should be created by SimulatorMaker
  theSimulatorImplementation = new LocalSimulatorImplementation();
}


