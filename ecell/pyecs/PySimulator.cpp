#include "PySimulator.h"
#include "StepCommand.h"

#include <string>

PySimulator::PySimulator()
{
  add_varargs_method("popCommand",&PySimulator::popCommand);//FOR TEST
  add_varargs_method("test",&PySimulator::test);//FOR TEST
  
  theSimulator = new Simulator();
}

void PySimulator::pushCommand( Command* command )
{
  cout<<"this is PySimulator::pushCommand method."<<endl;
  Simulator::pushCommand( command );
}

Py::Object PySimulator::popCommand( const Py::Tuple& args )
{
  Simulator::popCommand();
  return Py::Object();
}

Py::Object PySimulator::test( const Py::Tuple& args )
{
  cout<<"this is PySimulator::test method."<<endl;
  Command* command = new StepCommand();
  Simulator::pushCommand( command );
  return Py::Object();
}
