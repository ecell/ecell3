// Warning:
//
// the number of Substrate of DecayReactor must be one.
//

#include "libecs.hpp"
#include "Util.hpp"

#include "ContinuousProcess.hpp"


USE_LIBECS;

LIBECS_DM_CLASS( DecayFluxProcess, ContinuousProcess )
{

 public:

  LIBECS_DM_OBJECT( DecayFluxProcess, Process )
    {
      INHERIT_PROPERTIES( ContinuousProcess );

      PROPERTYSLOT_SET_GET( Real, T );
    }  

  DecayFluxProcess()
    :
    T( 1.0 ),
    k( 0.0 )
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
      
      k = log( 2.0 ) / T;
      S0 = getVariableReference( "S0" );
    }

  virtual void process()
    {
      Real velocity( k );

      velocity *= pow( S0.getValue(), - S0.getCoefficient() );
      setFlux( velocity );
    }

 protected:
  
  Real T;
  Real k;
  VariableReference S0;
  
};

LIBECS_DM_INIT( DecayFluxProcess, Process );
