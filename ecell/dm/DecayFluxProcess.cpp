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
      /*
      CLASS_INFO( BriefDescription, "A continuous decay process." );

      CLASS_INFO( Description, 
		  "DecayFluxProcess is a FluxProcess which calculates "
		  "mass-action decay process with a half-time T.\n\n"
		  "This is a mass-action reaction with a single reactant"
		  "VariableReference S0, which must be specified "
		  "in the model:\n"
		  "S0 --> (..0, 1, or more products..)\n"
		  "The half-time T is converted to the rate constant k as:\n"
		  "k = log( 2 ) / T\n\n"
		  "Flux rate of this Process is calculated by the following "
		  "equation:\n"
		  "flux rate = k * pow( S0.Value, S0.Coefficient )\n"
		  "When the coefficient of S0 is 1, then it is simplified as:"
		  "flux rate = k * S0.Value\n\n"
		  "Although only S0 is used for calculating the flux rate,"
		  "velocities are set to all VariableReferences with non-zero"
		  "coefficients, as defined in the FluxProcess base class.\n"
		  "Zero or negative half time is not allowed.\n" );

      CLASS_INFO( IsArbitraryPropertyAccepted,        "false" );

      CLASS_INFO( VariableReference__S0__Description,
		  "A Variable that decays." );
      CLASS_INFO( VariableReference__S0__Required,    "true" );
      */

      INHERIT_PROPERTIES( ContinuousProcess );

      PROPERTYSLOT_SET_GET( Real, T );

      /*
      CLASS_INFO( PropertySlot__T__BriefDescription, "half-time" );
      CLASS_INFO( PropertySlot__T__Description, 
		  "A positive, non-zero half-time in second." );
      CLASS_INFO( PropertySlot__T__Unit,        "sec" );
      CLASS_INFO( PropertySlot__T__Default,     "1.0" );
      CLASS_INFO( PropertySlot__T__IsRequired,  "false" );
      */

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

  virtual void fire()
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
