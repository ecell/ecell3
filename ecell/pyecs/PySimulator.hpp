#ifndef ___PY_SIMULATOR_H___
#define ___PY_SIMULATOR_H___

#include "Simulator.hpp"
#include "CXX/Extensions.hxx"

class PySimulator 
  :
  public Py::PythonExtension< PySimulator >,
  public Simulator
{

  Simulator* theSimulator;

public:
  
  PySimulator();
  ~PySimulator(){};

  static void init_type();

  PySimulator* getPySimulatorPtr() { return this; }
  Simulator* getSimulatorPtr() { return theSimulator; }

  Py::Object makePrimitive( const Py::Tuple& args );
  Py::Object sendMessage( const Py::Tuple& args );
  Py::Object getMessage( const Py::Tuple& args );
  Py::Object step( const Py::Tuple& args );

};

#endif   /* ___PY_SIMULATOR_H___ */








