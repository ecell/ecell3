#ifndef ___PY_SIMULATOR_H___
#define ___PY_SIMULATOR_H___

#include "Simulator.h"
#include "Command.h"
#include "CXX/Extensions.hxx"

class PySimulator 
  :
  public Py::PythonExtension<PySimulator>,
  public Simulator
{

  Simulator* theSimulator;

public:
  
  PySimulator();
  ~PySimulator(){};

  static void init_type();

  PySimulator* getPySimulatorPtr() { return this; }
  Simulator* getSimulatorPtr() { return theSimulator; }
  //  Py::Object pushCommand( const Py::Tuple& );
  void pushCommand( Command* );
  Py::Object popCommand( const Py::Tuple& );
  Py::Object test( const Py::Tuple& );

};

#endif   /* ___PY_SIMULATOR_H___ */








