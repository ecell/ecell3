#include <iostream>

#include "Simulator.hpp"

#include "PySimulator.hpp"

#include "PyEcs.hpp"

//-----------------//
// PyEcs class     //
//-----------------// 

#include <dlfcn.h>

PyEcs::PyEcs()
  :
  Py::ExtensionModule<PyEcs>( "ecs" )
{
  PySimulator::init_type();
  add_varargs_method( "Simulator", 
		      &PyEcs::createSimulator, 
		      "Simulator( type = \"Local\" )" );
  //  dlopen("./ecs.so",RTLD_NOW|RTLD_GLOBAL);

    
  initialize();
}

Object PyEcs::createSimulator( const Tuple& args )
{
  PySimulator* aPySimulator = new PySimulator();
  return asObject( aPySimulator );
}

void initecs()
{
  static PyEcs* ecs = new PyEcs();
}









