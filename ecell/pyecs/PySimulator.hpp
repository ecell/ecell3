#ifndef ___PY_SIMULATOR_H___
#define ___PY_SIMULATOR_H___

#include "emc/Simulator.hpp"
#include "CXX/Extensions.hxx"

using namespace Py;

class PySimulator 
  :
  public PythonExtension< PySimulator >,
  public Simulator
{
public:
  
  PySimulator();
  virtual ~PySimulator(){};

  static void init_type();

  Object makePrimitive( const Tuple& args );
  Object sendMessage( const Tuple& args );
  Object getMessage( const Tuple& args );
  Object step( const Tuple& args );

};

#endif   /* ___PY_SIMULATOR_H___ */








