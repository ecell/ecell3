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
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KcF );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KcR );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, Keq );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KmS0 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KmS1 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KmP0 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KmP1 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KiS0 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KiS1 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KiP0 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KiP1 );
    }
  
  SIMPLE_SET_GET_METHOD( Real, KcF );
  SIMPLE_SET_GET_METHOD( Real, KcR );
  SIMPLE_SET_GET_METHOD( Real, Keq );
  SIMPLE_SET_GET_METHOD( Real, KmS0 );
  SIMPLE_SET_GET_METHOD( Real, KmS1 );
  SIMPLE_SET_GET_METHOD( Real, KmP0 );
  SIMPLE_SET_GET_METHOD( Real, KmP1 );
  SIMPLE_SET_GET_METHOD( Real, KiS0 );
  SIMPLE_SET_GET_METHOD( Real, KiS1 );
  SIMPLE_SET_GET_METHOD( Real, KiP0 );
  SIMPLE_SET_GET_METHOD( Real, KiP1 );
    
  virtual void initialize()
    {
      FluxProcess::initialize();
      S0 = getVariableReference( "S0" );
      S1 = getVariableReference( "S1" );
      P0 = getVariableReference( "P0" );
      P1 = getVariableReference( "P1" );
      C0 = getVariableReference( "C0" );
    }

  virtual void process()
    {
      Real S0Concentration = S0.getVariable()->getConcentration();
      Real S1Concentration = S1.getVariable()->getConcentration();
      Real P0Concentration = P0.getVariable()->getConcentration();
      Real P1Concentration = P1.getVariable()->getConcentration();
      
      Real Denom( KcR * KiS0 * KmS1
		  + KcR * KmS1 * S0Concentration + KcR * KmS0 
		  * S1Concentration + KcF * KmP1 * P0Concentration / Keq
		  + KcF * KmP0 * P1Concentration / Keq + KcR 
		  * S0Concentration * S1Concentration
		  + KcF * KmP1 * S0Concentration * P0Concentration 
		  / (Keq * KiS0) + KcF * P0Concentration 
		  * P1Concentration / Keq
		  + KcR * KmS0 * S1Concentration * P1Concentration 
		  / KiP1 + KcR * S0Concentration * S1Concentration 
		  * P0Concentration /KiP0
		  + KcF * S1Concentration * P0Concentration 
		  * P1Concentration / (KiS1 * Keq) );
      
      Real velocity( KcF * KcR * C0.getVariable()->getValue()
		     * ( S0Concentration * S1Concentration 
			 - P0Concentration * P1Concentration / Keq ) 
		     / Denom );
      
      setFlux( velocity );
    }

 protected:

  Real KcF;
  Real KcR;
  Real Keq;

  Real KmS0;
  Real KmS1;
  Real KmP0;
  Real KmP1;

  Real KiS0;
  Real KiS1;
  Real KiP0;
  Real KiP1;

  VariableReference S0;
  VariableReference S1;
  VariableReference P0;
  VariableReference P1;
  VariableReference C0;
  
};

ECELL3_DM_INIT;
