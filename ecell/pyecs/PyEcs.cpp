#include <iostream>

#include "PyEcs.hpp"
#include "PySimulator.hpp"
#include "Simulator.hpp"

//-----------------//
// PyEcs class     //
//-----------------// 

PyEcs::PyEcs()
  :
  Py::ExtensionModule<PyEcs>("ecs")
{
  add_varargs_method( "makeSimulator", &PyEcs::makeSimulator );
  add_varargs_method( "makePrimitive", &PyEcs::makePrimitive );
  add_varargs_method( "sendMessage", &PyEcs::sendMessage );
  add_varargs_method( "getMessage", &PyEcs::getMessage );
  add_varargs_method( "step", &PyEcs::step );
  initialize();
}

Py::Object PyEcs::makeSimulator( const Py::Tuple& args )
{
  cout<<"this is PyEcs::makeSimulator module."<<endl;
  PySimulator* pySimulator = new PySimulator();
  return Py::asObject( pySimulator );
}

Py::Object PyEcs::step( const Py::Tuple& args )
{
  cout<<"this is PyEcs::step module."<<endl;
  Simulator* simulator = ((PySimulator&)args[0]).getSimulatorPtr();
  simulator->step();

  return Py::Object();
}

Py::Object PyEcs::makePrimitive( const Py::Tuple& args )
{
  cout<<"this is PyEcs::makePrimitive module."<<endl;

  const string classname = (string)(Py::String)args[1];
  const string fqen = (string)(Py::String)args[2];
  const string name = (string)(Py::String)args[3];

  //  PySimulator& simulator = (PySimulator&)args[0]; 

  return Py::Object();
}
  
Py::Object PyEcs::sendMessage( const Py::Tuple& args )
{
  cout<<"this is PyEcs::sendMessage module."<<endl;
  return Py::Object();
}

Py::Object PyEcs::getMessage( const Py::Tuple& args )
{
  cout<<"this is PyEcs::getMessage module."<<endl;
  return Py::Object();
}

void initecs()
{
  static PyEcs* ecs = new PyEcs();
}









