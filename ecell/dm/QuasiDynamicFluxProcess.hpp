#include "libecs.hpp"
#include "ContinuousProcess.hpp"
#include "Util.hpp"
#include "FullID.hpp"
#include "PropertyInterface.hpp"

#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "Interpolant.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( QuasiDynamicFluxProcess, ContinuousProcess )
{

 public:

  LIBECS_DM_OBJECT( QuasiDynamicFluxProcess, Process )
    {
      INHERIT_PROPERTIES( Process );
      PROPERTYSLOT_SET_GET( Integer, Irreversible );
      PROPERTYSLOT_SET_GET( Real, Vmax );
      PROPERTYSLOT_SET_GET( Polymorph, FluxDistributionList );
    }

  QuasiDynamicFluxProcess()
    :    
    Irreversible( 0 ),
    Vmax( 0 )
    {
      theFluxDistributionVector.reserve( 0 );
    }

  ~QuasiDynamicFluxProcess()
    {
      ; // do nothing
    }

  SIMPLE_SET_GET_METHOD( Integer, Irreversible );
  SIMPLE_SET_GET_METHOD( Real, Vmax );

  SET_METHOD( Polymorph, FluxDistributionList )
    {
      const PolymorphVector aVector( value.asPolymorphVector() );
      
      theFluxDistributionVector.clear();
      for( PolymorphVectorConstIterator i( aVector.begin() );
	   i != aVector.end(); ++i )
	{
	  theFluxDistributionVector.push_back( ( *( findVariableReference( (*i).asString() ) ) ) );
	}      
    }
  
  GET_METHOD( Polymorph, FluxDistributionList )
    {
      PolymorphVector aVector;
      for( VariableReferenceVectorConstIterator
	     i( theFluxDistributionVector.begin() );
	   i != theFluxDistributionVector.end() ; ++i )
	{
	  FullID aFullID( (*i).getVariable()->getFullID() );
	  aVector.push_back( aFullID.getString() );
	}

      return aVector;
    }

  VariableReferenceVector getFluxDistributionVector()
    {
      return theFluxDistributionVector;
    }

  virtual void initialize()
    {
      Process::initialize();      
      if( theFluxDistributionVector.empty() )
	{
	  theFluxDistributionVector = theVariableReferenceVector;
	} 
    }

  virtual void fire()
    {
      ; // do nothing
    }
  
 protected:

  VariableReferenceVector theFluxDistributionVector;
  Integer Irreversible;
  Real Vmax;

};


