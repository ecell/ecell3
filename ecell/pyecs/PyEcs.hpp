#ifndef ___PYECS_H___
#define ___PYECS_H___

#include "CXX/Extensions.hxx"

class PyEcs
  : 
  public Py::ExtensionModule< PyEcs >
{

public:  

  PyEcs();
  ~PyEcs(){};
 
  static void init_type();
  
  Py::Object makeSimulator( const Py::Tuple& args );

private:

};   //end of class PyEcs

extern "C" void initecs();

#endif   /* ___PYECS_H___ */













