#ifndef ___PY_COMMAND_H___
#define ___PY_COMMAND_H___

#include "CXX/Extensions.hxx"

class PyCommand 
  :
  public Py::PythonExtension<PyCommand>
{

public:
  
  PyCommand();
  ~PyCommand(){};

  static void init_type();

};

#endif   /* ___PY_COMMAND_H___ */








