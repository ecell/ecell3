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

      theReactantQuantitySlotVector.clear();
      theReactantVelocitySlotVector.clear();

      for( ReactantMapIterator s( theReactantMap.begin() );
	   s != theReactantMap.end() ; ++s )
	{
	  SubstancePtr aSubstance( s->second.getSubstance() );

	  theReactantQuantitySlotVector.push_back( aSubstance->
						   getPropertySlot( "Quantity", this ) );
	  theReactantVelocitySlotVector.push_back( aSubstance->
						   getPropertySlot( "Velocity", this ) );

	}

    }

    void process( const Real velocity )
    {
      Real aVelocityPerStep( velocity );
      aVelocityPerStep *= getSuperSystem()->getStepper()->getStepInterval();

      // The aVelocityPerStep is limited by amounts of substrates and products.
#if 0
      if( aVelocityPerStep >= 0 )  // If the reaction occurs forward,
	{
	  for( ReactantMapIterator s( theReactantMap.begin() );
	       s != theReactantMap.end() ; ++s)
	    {
	      Real aLimit( (*s)->getSubstance().getQuantity()
			   * (*s)->getStoichiometry() );
	      if( aVelocityPerStep > aLimit )
		{
		  aVelocityPerStep = aLimit;
		}
	    }
	}
      else  // Or if the reaction occurs reversely
	{
	  for( ReactantVectorIterator p( theProductList.begin() );
	       p != theProductList.end() ; ++p)
	    {
	      Real aLimit( - (*p)->getSubstance().getQuantity()
			   * (*p)->getStoichiometry() );
	      if( aVelocityPerStep < aLimit )
		{
		  aVelocityPerStep = aLimit;
		}
	    }          
	}
#endif
    // IMPORTANT!!!: 
    // Activity must be given as 
    // [number of molecule that this reactor yields / deltaT]
    // *BUT* this is *RECALCULATED* as [number of molecule / s]
    // in Reactor::activity().
    setActivity( aVelocityPerStep );

    // Increase or decrease reactants.

    PropertySlotVectorIterator i( theReactantVelocitySlotVector.begin() );
    for( ReactantMapIterator s( theReactantMap.begin() );
	 s != theReactantMap.end() ; ++s )
      {
	(*i)->setReal( aVelocityPerStep * (*s).second.getStoichiometry() );
	++i;
      }

    }

  protected:

    PropertySlotVector theReactantQuantitySlotVector;
    PropertySlotVector theReactantVelocitySlotVector;
  };

}

#endif /* __FluxReactor_HPP */







