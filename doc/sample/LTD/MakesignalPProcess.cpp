#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"
#include "PropertySlotMaker.hpp"
#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "VariableProxy.hpp"

#include "ecell3_dm.hpp"

#define ECELL3_DM_TYPE Process

USE_LIBECS;

ECELL3_DM_CLASS
  :  
  public Process
{
  
  ECELL3_DM_OBJECT;
  
 public:
  
  ECELL3_DM_CLASSNAME()
    {
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, add );
    }
  
  SIMPLE_SET_GET_METHOD( Real, add );
  
  void initialize()
    {
      Process::initialize();
      P0 = getVariableReference( "P0" );
      i = 0;
      k = 0;
    }

  void process()
  {
    Real p0( P0.getValue() );
    
    i += 1;
    
    if( k < 30000)
      {
	Int ii( i % 10 );
	if( ii == 0 )
	  {
	    p0 = add * getSuperSystem()->getSizeN_A();
	    k = k + 1;
	  }
	else
	  {
	    p0 = add * getSuperSystem()->getSizeN_A() / (2 * ii);
	  }
      }
    else
      {
	p0 = 0.00000000000000000001 * getSuperSystem()->getSizeN_A();
      }					
    P0.setValue(p0);
  }
  
 protected:

  Real add;
  Int i;
  Int k;
  
  VariableReference P0;  
  
};

ECELL3_DM_INIT;
