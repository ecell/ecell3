#include "libecs.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"

#include "System.hpp"

#include "ContinuousProcess.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( MassActionFluxProcess, ContinuousProcess )
{

 public:

  LIBECS_DM_OBJECT( MassActionFluxProcess, Process )
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( Real, k );
    }

  MassActionFluxProcess()
    :
    k( 0.0 )
    {
      ; // do nothing
    }
  
  SIMPLE_SET_GET_METHOD( Real, k );
  
  virtual void fire()
  {
    
    Real velocity( k * N_A );
    velocity *= getSuperSystem()->getSize();

    for( VariableReferenceVectorConstIterator 
	   s( theVariableReferenceVector.begin() );
	 s != theZeroVariableReferenceIterator; ++s )
      {
	VariableReference aVariableReference( *s );
	Integer aCoefficient( aVariableReference.getCoefficient() );
	do {
	  ++aCoefficient;
	  velocity *= aVariableReference.getMolarConc();
	} while( aCoefficient != 0 );
	
      }
    
    setFlux(velocity);
    
  }
  
  virtual void initialize()
  {
    Process::initialize();
    declareUnidirectional();
  }  

 protected:
  
  Real k;
    
};

LIBECS_DM_INIT( MassActionFluxProcess, Process );
