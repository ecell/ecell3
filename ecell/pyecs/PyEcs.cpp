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
  add_varargs_method( "createSimulator", &PyEcs::createSimulator );
  initialize();
}

Py::Object PyEcs::createSimulator( const Py::Tuple& args )
{
  cout<<"this is PyEcs::createSimulator module."<<endl;
  PySimulator* pySimulator = new PySimulator();
  return Py::asObject( pySimulator );
}

void initecs()
{
  static PyEcs* ecs = new PyEcs();
}









