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
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KmS1 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KmS2 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KmP1 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KmP2 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KiS1 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KiS2 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KiP1 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KiP2 );
    }
  
  SIMPLE_SET_GET_METHOD( Real, KcF );
  SIMPLE_SET_GET_METHOD( Real, KcR );
  SIMPLE_SET_GET_METHOD( Real, Keq );
  SIMPLE_SET_GET_METHOD( Real, KmS1 );
  SIMPLE_SET_GET_METHOD( Real, KmS2 );
  SIMPLE_SET_GET_METHOD( Real, KmP1 );
  SIMPLE_SET_GET_METHOD( Real, KmP2 );
  SIMPLE_SET_GET_METHOD( Real, KiS1 );
  SIMPLE_SET_GET_METHOD( Real, KiS2 );
  SIMPLE_SET_GET_METHOD( Real, KiP1 );
  SIMPLE_SET_GET_METHOD( Real, KiP2 );
    
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
      
      Real Denom( KcR * KiS1 * KmS2
		  + KcR * KmS2 * S0Concentration + KcR * KmS1 
		  * S1Concentration + KcF * KmP2 * P0Concentration / Keq
		  + KcF * KmP1 * P1Concentration / Keq + KcR 
		  * S0Concentration * S1Concentration
		  + KcF * KmP2 * S0Concentration * P0Concentration 
		  / (Keq * KiS1) + KcF * P0Concentration 
		  * P1Concentration / Keq
		  + KcR * KmS1 * S1Concentration * P1Concentration 
		  / KiP2 + KcR * S0Concentration * S1Concentration 
		  * P0Concentration /KiP1
		  + KcF * S1Concentration * P0Concentration 
		  * P1Concentration / (KiS2 * Keq) );
      
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

  Real KmS1;
  Real KmS2;
  Real KmP1;
  Real KmP2;

  Real KiS1;
  Real KiS2;
  Real KiP1;
  Real KiP2;

  VariableReference S0;
  VariableReference S1;
  VariableReference P0;
  VariableReference P1;
  VariableReference C0;
  
};

ECELL3_DM_INIT;
