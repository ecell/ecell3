#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "FullID.hpp"
#include "PropertyInterface.hpp"

#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "VariableProxy.hpp"


USE_LIBECS;

LIBECS_DM_CLASS( QuasiDynamicFluxProcess, Process )
{

 public:

  LIBECS_DM_OBJECT( QuasiDynamicFluxProcess, Process )
    {
      INHERIT_PROPERTIES( Process );
    }

  QuasiDynamicFluxProcess()
    :
    {
      ; // do nothing
    }

  ~QuasiDynamicFluxProcess()
    {
      ; // do nothing
    }
  

  virtual void initialize()
    {
      Process::initialize();
      declareUnidirectional();
    }

  virtual void fire()
  {
    ; // do nothing
  }
  
 protected:

};


