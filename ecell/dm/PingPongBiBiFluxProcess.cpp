#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"
#include "PropertySlotMaker.hpp"
#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "VariableProxy.hpp"

#include "Process.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( PingPongBiBiProcess, Process )
{

 public:

  LIBECS_DM_OBJECT( PingPongBiBiProcess, Process )
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( Real, KcF );
      PROPERTYSLOT_SET_GET( Real, KcR );
      PROPERTYSLOT_SET_GET( Real, Keq );
      PROPERTYSLOT_SET_GET( Real, KmS0 );
      PROPERTYSLOT_SET_GET( Real, KmS1 );
      PROPERTYSLOT_SET_GET( Real, KmP0 );
      PROPERTYSLOT_SET_GET( Real, KmP1 );
      PROPERTYSLOT_SET_GET( Real, KiS0 );
      PROPERTYSLOT_SET_GET( Real, KiP1 );
    }

  // FIXME: initial values
  PingPongBiBiProcess()
    {
      ; // do nothing
    }
  
  SIMPLE_SET_GET_METHOD( Real, KcF );
  SIMPLE_SET_GET_METHOD( Real, KcR );
  SIMPLE_SET_GET_METHOD( Real, Keq );
  SIMPLE_SET_GET_METHOD( Real, KmS0 );
  SIMPLE_SET_GET_METHOD( Real, KmS1 );
  SIMPLE_SET_GET_METHOD( Real, KmP0 );
  SIMPLE_SET_GET_METHOD( Real, KmP1 );
  SIMPLE_SET_GET_METHOD( Real, KiS0 );
  SIMPLE_SET_GET_METHOD( Real, KiP1 );
    
  virtual void initialize()
    {
      Process::initialize();
      S0 = getVariableReference( "S0" );
      S1 = getVariableReference( "S1" );
      P0 = getVariableReference( "P0" );
      P1 = getVariableReference( "P1" );
      C0 = getVariableReference( "C0" );
    }

  virtual void process()
    {
      Real S0Concentration = S0.getConcentration();
      Real S1Concentration = S1.getConcentration();
      Real P0Concentration = P0.getConcentration();
      Real P1Concentration = P1.getConcentration();
      
      Real Denom =
	+ KcR * KmS1 * S0Concentration 
	+ KcR * KmS0 * S1Concentration 
	+ KcF * KmP1 * P0Concentration / Keq
	+ KcF * KmP0 * P1Concentration / Keq 
	+ KcR * S0Concentration * S1Concentration
	+ KcF * KmP1 * S0Concentration * P0Concentration / (Keq * KiS0) 
	+ KcF * P0Concentration * P1Concentration / Keq
	+ KcR * KmS0 * S1Concentration * P1Concentration / KiP1;
      
      
      Real velocity = KcF * KcR * C0.getValue()
	* ( S0Concentration * S1Concentration 
	    - P0Concentration * P1Concentration / Keq ) / Denom;

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
  Real KiP1;

  VariableReference S0;
  VariableReference S1;
  VariableReference P0;
  VariableReference P1;
  VariableReference C0;
  
};

LIBECS_DM_INIT( PingPongBiBiProcess, Process );
