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
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KmP );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KiS );
    }
  
  SIMPLE_SET_GET_METHOD( Real, KcF );
  SIMPLE_SET_GET_METHOD( Real, KcR );
  SIMPLE_SET_GET_METHOD( Real, Keq );
  SIMPLE_SET_GET_METHOD( Real, KmS0 );
  SIMPLE_SET_GET_METHOD( Real, KmS1 );
  SIMPLE_SET_GET_METHOD( Real, KmP );
  SIMPLE_SET_GET_METHOD( Real, KiS );
    
  virtual void initialize()
    {
      FluxProcess::initialize();
      S0 = getVariableReference( "S0" );
      S1 = getVariableReference( "S1" );
      P0 = getVariableReference( "P0" );
      C0 = getVariableReference( "C0" );
    }

  virtual void process()
    {
      Real S0Concentration = S0.getVariable()->getConcentration();
      Real S1Concentration = S1.getVariable()->getConcentration();
      Real P0Concentration = P0.getVariable()->getConcentration();
      
      Real Den( KcF * KmP + KcF * P0Concentration 
		+ KcR * KmS0 * S1Concentration / Keq
		+ KcR * KmS1 * S0Concentration / Keq 
		+ KcF * P0Concentration * S1Concentration / KiS
		+ KcF * S0Concentration * S1Concentration / Keq );
      
      Real velocity = KcF * KcR * C0.getVariable()->getValue()
	* (P0Concentration  - S0Concentration  * S1Concentration  / Keq) / Den;    
      setFlux( velocity );
    }

 protected:

  Real KcF;
  Real KcR;
  Real Keq;

  Real KmS0;
  Real KmS1;
  Real KmP;

  Real KiS;

  VariableReference S0;
  VariableReference S1;
  VariableReference P0;
  VariableReference C0;
  
};

ECELL3_DM_INIT;
