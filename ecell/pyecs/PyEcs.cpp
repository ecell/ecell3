#include <iostream>

#include "PyEcs.h"
#include "PySimulator.h"
#include "Simulator.h"
#include "StepCommand.h"
#include "MakeSystemCommand.h"

//-----------------//
// PyEcs class     //
//-----------------// 

PyEcs::PyEcs()
  :
  Py::ExtensionModule<PyEcs>("ecs")
{
  add_varargs_method( "simulator", &PyEcs::simulator );
  add_varargs_method( "step", &PyEcs::step );
  add_varargs_method( "makeSystem", &PyEcs::makeSystem );
  initialize();
}

Py::Object PyEcs::simulator( const Py::Tuple& args )
{
  cout<<"this is PyEcs::simulator module."<<endl;
  PySimulator* simulator = new PySimulator();
  return Py::asObject( simulator );
}

Py::Object PyEcs::step( const Py::Tuple& args )
{
  cout<<"this is PyEcs::step module."<<endl;

  //  Command* command = new StepCommand();
  //  simulator.pushCommand( command );
  //  simulator.Simulator::pushCommand( command );
  //(PySimulator&)args[0].test( Py::Tuple() );
  //  PySimulator& simulator = ( PySimulator& )args[0];
  //  simulator.test( Py::Tuple() );
  //  ( ( PySimulator& )args[0] ).test( Py::Tuple() );
  //  simulator.test( Py::Tuple() );
  //  PySimulator* simulator = ((PySimulator&)args[0]).getPySimulatorPtr();
  Simulator* simulator = ((PySimulator&)args[0]).getSimulatorPtr();
  simulator->test();

  return Py::Object();
}

Py::Object PyEcs::makeSystem( const Py::Tuple& args )
{
  cout<<"this is PyEcs::makeSystem module."<<endl;

  const string classname = (string)(Py::String)args[1];
  const string fqen = (string)(Py::String)args[2];
  const string name = (string)(Py::String)args[3];

  PySimulator& simulator = (PySimulator&)args[0]; 
  Command* command = new MakeSystemCommand( classname, fqen, name );
  simulator.pushCommand( command );

  return Py::Object();
}
  
void initecs()
{
  static PyEcs* ecs = new PyEcs();
}









