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

#include "Process.hpp"


USE_LIBECS;

LIBECS_DM_CLASS( DecayProcess, Process )
{

 public:

  LIBECS_DM_OBJECT( DecayProcess, Process )
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( Real, T );
    }  

  DecayProcess()
    {
      ; // do nothing
    }

  
  SIMPLE_SET_GET_METHOD( Real, T );
    
  virtual void initialize()
    {
      Process::initialize();

      if(T <= 0)
	{
	  THROW_EXCEPTION( ValueError, "Error:in DecayProcess::initialze().0 or negative half time. set to 1." );
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

LIBECS_DM_INIT( DecayProcess, Process );
