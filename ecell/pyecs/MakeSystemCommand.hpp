#ifndef ___MAKESYSTEM_COMMAND_H___
#define ___MAKESYSTEM_COMMAND_H___

#include "Command.h"

#include "Koyurugi/RootSystem.h"

#include <string>

class MakeSystemCommand : public Command
{

  string theClassname;
  string theFqen;
  string theName;

public:

  MakeSystemCommand( const string&, 
		     const string&, 
		     const string& );

  ~MakeSystemCommand() {};

  void doit();

};   //end of class MakeSystemCommand
    
#endif   /* ___MAKESYSTEM_COMMAND_H___ */






