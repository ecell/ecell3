#include "LocalSimulatorImplementation.cpp"

#include "Simulator.hpp"

Simulator::Simulator()
{
  theSimulatorImplementation = new LocalSimulatorImplementation();
}


