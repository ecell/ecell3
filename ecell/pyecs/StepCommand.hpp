#ifndef ___STEP_COMMAND_H___
#define ___STEP_COMMAND_H___

#include "Command.h"

#include "Koyurugi/RootSystem.h"


class StepCommand 
  : 
  public Command
{

public:

  StepCommand();
  ~StepCommand() {};

  void doit();

};   //end of class StepCommand
    
#endif   /* ___STEP_COMMAND_H___ */






