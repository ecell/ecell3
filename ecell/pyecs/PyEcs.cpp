#include <iostream>

#include "PySimulator.hpp"
#include "Simulator.hpp"

#include "PyEcs.hpp"

//-----------------//
// PyEcs class     //
//-----------------// 

PyEcs::PyEcs()
  :
  Py::ExtensionModule<PyEcs>( "ecs" )
{
  PySimulator::init_type();
  add_varargs_method( "Simulator", 
		      &PyEcs::createSimulator, 
		      "Simulator( type = \"Local\" )" );
    
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









