#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"

#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "Interpolant.hpp"

#include "Process.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( CatalyzedMassActionProcess, Process )
{

 public:

  LIBECS_DM_OBJECT( CatalyzedMassActionProcess, Process )
    {
      INHERIT_PROPERTIES( Process );
      PROPERTYSLOT_SET_GET( Real, k );
    }


  CatalyzedMassActionProcess()
    {
      ; // do nothing
    }
  
  SIMPLE_SET_GET_METHOD( Real, k );
    
  virtual void initialize()
    {
      Process::initialize();

      C0 = getVariableReference( "C0" );
    }

  virtual void fire()
    {
      Real velocity( k );
      velocity *= C0.getValue();
      for( VariableReferenceVectorConstIterator
	     i ( theVariableReferenceVector.begin() );
           i != theZeroVariableReferenceIterator ; ++i )
	{
	  VariableReferenceCref aVariableReference( *i );
	  const Real aConcentration( aVariableReference.getMolarConc() );

	  Integer j( aVariableReference.getCoefficient() );
	  do
	    {	   
	      velocity *= aConcentration;
	      ++j;
	    } while( j != 0 );
	}

      setFlux( velocity );
    }
  
 protected:
  
  Real k;
  VariableReference C0;

};

LIBECS_DM_INIT( CatalyzedMassActionProcess, Process );
