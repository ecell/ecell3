#ifndef ___PYECS_HPP___
#define ___PYECS_HPP___

#include "CXX/Extensions.hxx"

class PyEcs
  : 
  public Py::ExtensionModule< PyEcs >
{

public:  

  PyEcs();
  ~PyEcs(){};
 
  static void init_type();
  
  Py::Object simulator( const Py::Tuple& args );
  Py::Object step( const Py::Tuple& args );
  Py::Object makeSystem( const Py::Tuple& args );

private:

};   //end of class PyEcs

extern "C" void initecs();

#endif   /* ___PYECS_HPP___ */













