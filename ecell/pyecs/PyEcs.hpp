#ifndef ___PYECS_H___
#define ___PYECS_H___

#include "CXX/Extensions.hxx"

using namespace Py;

class PyEcs
  : 
  public ExtensionModule< PyEcs >
{

public:  

  PyEcs();
  ~PyEcs(){};
 
  Object createSimulator( const Tuple& args );

private:

};   //end of class PyEcs

extern "C" void initecs();

#endif   /* ___PYECS_H___ */













