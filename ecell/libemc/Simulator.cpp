#include "Simulator.h"
#include "StepCommand.h"

Simulator::Simulator()
{
  theSimulatorImplementation = new SimulatorImplementation();
}

void Simulator::pushCommand( Command* command )
{
  theSimulatorImplementation->pushCommand( command );
}

void Simulator::popCommand()
{
  theSimulatorImplementation->popCommand();
}

void Simulator::test()
{
  Command* command = new StepCommand();
  theSimulatorImplementation->pushCommand( command );
}






