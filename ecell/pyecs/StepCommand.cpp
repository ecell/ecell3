#include "StepCommand.h"
#include "Koyurugi/Stepper.h"

StepCommand::StepCommand()
{
}

void StepCommand::doit()
{
  cout<<"this is StepCommand::doit()."<<endl;
  theRootSystem->getStepperLeader().step();
  cout<<"check"<<endl;
}
