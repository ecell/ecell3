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
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KmS );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KmP );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KcF );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, Keq );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KiiP );
    }
  
  SIMPLE_SET_GET_METHOD( Real, KmS );
  SIMPLE_SET_GET_METHOD( Real, KmP );
  SIMPLE_SET_GET_METHOD( Real, KcF );
  SIMPLE_SET_GET_METHOD( Real, Keq );
  SIMPLE_SET_GET_METHOD( Real, KiiP );
    
  virtual void initialize()
    {
      FluxProcess::initialize();
      S0 = getVariableReference( "S0" );
      P0 = getVariableReference( "P0" );
      C0 = getVariableReference( "C0" );  
    }

  virtual void process()
    {
      Real S( S0.getVariable()->getConcentration() );
      Real P( P0.getVariable()->getConcentration() );
      Real velocity( KcF * C0.getVariable()->getConcentration() * 
	(S-P/Keq) / KmS * (1+P/KmP) + S * (1+P/KiiP) );

      setFlux( velocity );
    }

 protected:
  

  Real KmS;
  Real KmP;
  Real KcF;
  Real Keq;
  
  Real KiiP;

  VariableReference S0;
  VariableReference P0;
  VariableReference C0;
  
};

ECELL3_DM_INIT;
