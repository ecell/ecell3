#include "SimulatorImplementation.h"

RootSystem* theRootSystem = new RootSystem();

SimulatorImplementation::SimulatorImplementation()
{
}

void SimulatorImplementation::pushCommand( Command* command )
{
  cout << "this is SimulatorImplementation::pushCommand()." << endl;
  theCommandQueue.push( command );
}

void SimulatorImplementation::popCommand()
{
  cout << "this is SimulatorImplementation::popCommand()." << endl;
  theCommandQueue.front()->setRootSystem( getRootSystemPtr() );
  theCommandQueue.front()->doit();
  theCommandQueue.pop();
}

