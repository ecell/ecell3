// Warning:
//
// the number of Substrate of DecayReactor must be one.
//

#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"

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

      if( T <= 0.0 )
	{
	  THROW_EXCEPTION( InitializationFailed, 
			   "Zero or negative half time." );
	}
      
      k_na = N_A * log( 2.0 ) / T;
      S0 = getVariableReference( "S0" );
    }

  virtual void process()
    {
      Real velocity( k_na );
      velocity *= getSuperSystem()->getSize();

      velocity *= pow( S0.getMolarConc(), - S0.getCoefficient() );
      setFlux( velocity );
    }

 protected:
  
  Real T;
  Real k_na;
  VariableReference S0;
  
};

LIBECS_DM_INIT( DecayProcess, Process );
