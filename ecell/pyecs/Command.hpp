#ifndef ___COMMAND_H___
#define ___COMMAND_H___

#include "Koyurugi/RootSystem.h"

class Command
{

protected:
  
  RootSystem* theRootSystem;

public:
  
  Command();
  ~Command() {};
  
  void setRootSystem( RootSystem* );
  virtual void doit() = 0;

};   //end of class Command
    
#endif   /* ___COMMAND_H___ */







