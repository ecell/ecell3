#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"
#include "PropertySlotMaker.hpp"
#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "VariableProxy.hpp"

#include "FluxProcess.hpp"
#include "ecell3_dm.hpp"

#define ECELL3_DM_TYPE Process

USE_LIBECS;

ECELL3_DM_CLASS
  :  
  public FluxProcess
{

  ECELL3_DM_OBJECT;
  
 public:

  ECELL3_DM_CLASSNAME()
    {
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, k );
    }
  
  SIMPLE_SET_GET_METHOD( Real, k );
    
  virtual void initialize()
    {
      FluxProcess::initialize();
    }

  virtual void process()
    {

      Real velocity( k * N_A );
      velocity *= getSuperSystem()->getVolume();
      velocity *= C0.getVariable()->getValue();
      for( VariableReferenceVectorConstIterator
	     i ( theVariableReferenceVector.begin() );
           i != theZeroVariableReferenceIterator ; ++i )
	{
	  
	  for( Int j( (*i).getCoefficient()); j!=0; j-- )
	    {	   
	      velocity *= (*i).getVariable()->getConcentration();
	    }
	}

      setFlux( velocity );
    
    }
  
 protected:
  
  Real k;
  VariableReference C0;

};

ECELL3_DM_INIT;
