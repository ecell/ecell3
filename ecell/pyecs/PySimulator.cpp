#include "PySimulator.hpp"

PySimulator::PySimulator()
{
  add_varargs_method( "makePrimitive", &PySimulator::makePrimitive );
  add_varargs_method( "sendMessage", &PySimulator::sendMessage );
  add_varargs_method( "getMessage", &PySimulator::getMessage );
  add_varargs_method( "step", &PySimulator::step );

  theSimulator = new Simulator();
}

Py::Object PySimulator::step( const Py::Tuple& args )
{
  cout<<"this is PySimulator::step module."<<endl;
  theSimulator->step();

  return Py::Object();
}

Py::Object PySimulator::makePrimitive( const Py::Tuple& args )
{
  cout<<"this is PySimulator::makePrimitive module."<<endl;

  const string classname = (string)(Py::String)args[1];
  const string fqen = (string)(Py::String)args[2];
  const string name = (string)(Py::String)args[3];
  theSimulator->makePrimitive()

  return Py::Object();
}
  
Py::Object PySimulator::sendMessage( const Py::Tuple& args )
{
  cout<<"this is PySimulator::sendMessage module."<<endl;
  return Py::Object();
}

Py::Object PySimulator::getMessage( const Py::Tuple& args )
{
  cout<<"this is PySimulator::getMessage module."<<endl;
  return Py::Object();
}



