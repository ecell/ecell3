#ifndef __FLUXPROCESS_HPP
#define __FLUXPROCESS_HPP

#include "libecs/libecs.hpp"
#include "libecs/Process.hpp"
#include "libecs/Connection.hpp"
#include "libecs/Variable.hpp"
#include "libecs/Stepper.hpp"

namespace libecs
{

  class FluxProcess         
    :  
    public Process
  {
  
  public:

    FluxProcess()
    {
      ; // do nothing
    }

    ~FluxProcess()
    {
      ; // do nothing
    }

    StringLiteral getClassName() const { return "FluxProcess"; }
    
    void initialize()
    {
      Process::initialize();

    }

    void setFlux( RealCref velocity )
    {
      Real aVelocity( velocity );

      // The aVelocityPerStep is limited by amounts of substrates and products.
      /*
      for( ConnectionMapIterator s( theConnectionMap.begin() );
	   s != theConnectionMap.end() ; ++s )
	{
	  Connection aConnection( s->second );
	  Int aCoefficient( aConnection.getCoefficient() );
	  
	  if( aCoefficient != 0 )
	    {
	      Real aVelocityPerStep = velocity * aConnection.getVariable()->
		getStepper()->getStepInterval();

	      Real aLimit( aConnection.getVariable()->getValue() 
			   / aCoefficient );
	      if( ( aLimit > 0 && aVelocityPerStep > aLimit ) ||
		  ( aLimit < 0 && aVelocityPerStep < aLimit ) )
	      {
		aVelocityPerStep = aLimit;
	      }
	    }
	}
      */

    // IMPORTANT!!!: 
    // Activity must be given as 
    // [number of molecule that this process yields / deltaT]
    setActivity( aVelocity );

    // Increase or decrease connections.

    for( ConnectionMapIterator s( theConnectionMap.begin() );
	 s != theConnectionMap.end() ; ++s )
      {
	Connection aConnection( s->second );
	const Int aCoefficient( aConnection.getCoefficient() );
	if( aCoefficient != 0 )
	  {
	    aConnection.getVariable()->
	      addVelocity( aVelocity * aCoefficient );
	  }
      }

    }

  };

}

#endif /* __FluxProcess_HPP */







