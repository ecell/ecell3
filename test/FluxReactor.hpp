#ifndef __FLUXREACTOR_HPP
#define __FLUXREACTOR_HPP

#include "libecs/libecs.hpp"
#include "libecs/Reactor.hpp"
#include "libecs/Reactant.hpp"
#include "libecs/Substance.hpp"
#include "libecs/Stepper.hpp"

namespace libecs
{

  class FluxReactor         
    :  
    public Reactor
  {
  
  public:

    FluxReactor()
    {
      ; // do nothing
    }

    ~FluxReactor()
    {
      ; // do nothing
    }

    StringLiteral getClassName() const { return "FluxReactor"; }
    
    void initialize()
    {
      Reactor::initialize();

    }

    void process( RealCref velocity )
    {
      Real aVelocity( velocity );

      // The aVelocityPerStep is limited by amounts of substrates and products.
      /*
      for( ReactantMapIterator s( theReactantMap.begin() );
	   s != theReactantMap.end() ; ++s )
	{
	  Reactant aReactant( s->second );
	  Int aStoichiometry( aReactant.getStoichiometry() );
	  
	  if( aStoichiometry != 0 )
	    {
	      Real aVelocityPerStep = velocity * aReactant.getSubstance()->
		getStepper()->getStepInterval();

	      Real aLimit( aReactant.getSubstance()->getQuantity() 
			   / aStoichiometry );
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
    // [number of molecule that this reactor yields / deltaT]
    setActivity( aVelocity );

    // Increase or decrease reactants.

    for( ReactantMapIterator s( theReactantMap.begin() );
	 s != theReactantMap.end() ; ++s )
      {
	Reactant aReactant( s->second );
	const Int aStoichiometry( aReactant.getStoichiometry() );
	if( aStoichiometry != 0 )
	  {
	    aReactant.getSubstance()->
	      addVelocity( aVelocity * aStoichiometry );
	  }
      }

    }

  };

}

#endif /* __FluxReactor_HPP */







