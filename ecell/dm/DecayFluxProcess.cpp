// Warning:
//
// the number of Substrate of DecayReactor must be one.
//

#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"
#include "PropertySlotMaker.hpp"
#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "VariableProxy.hpp"

#include "FluxProcess.hpp"
#include "ecell3_dm.hpp"

#define ECELL3_DM_TYPE Process

USE_LIBECS;

ECELL3_DM_CLASS
  :  
  public FluxProcess
{

  ECELL3_DM_OBJECT;
  
 public:

  ECELL3_DM_CLASSNAME()
    {
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, T );
    }
  
  SIMPLE_SET_GET_METHOD( Real, T );
    
  virtual void initialize()
    {
      FluxProcess::initialize();

      if(T <= 0)
	{
	  THROW_EXCEPTION( ValueError, "Error:in DecayFluxProcess::initialze().0 or negative half time. set to 1." );
	}
      
      k = log(2)/T;
      S0 = getVariableReference( "S0" );
    }

  virtual void process()
    {
      Real velocity( k * N_A );
      velocity *= getSuperSystem()->getSize();

      velocity *= pow(S0.getConcentration(),S0.getCoefficient()*-1);
      setFlux( velocity );
    }

 protected:
  
  Real T;
  Real k;
  VariableReference S0;
  
};

ECELL3_DM_INIT;
