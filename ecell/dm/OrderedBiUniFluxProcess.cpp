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

LIBECS_DM_CLASS( OrderedBiUniProcess, Process )
{

 public:

  LIBECS_DM_OBJECT( OrderedBiUniProcess, Process )
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( Real, KcF );
      PROPERTYSLOT_SET_GET( Real, KcR );
      PROPERTYSLOT_SET_GET( Real, Keq );
      PROPERTYSLOT_SET_GET( Real, KmS0 );
      PROPERTYSLOT_SET_GET( Real, KmS1 );
      PROPERTYSLOT_SET_GET( Real, KmP );
      PROPERTYSLOT_SET_GET( Real, KiS );
    }


  // FIXME: initial values?
  OrderedBiUniProcess()
    {
      ; // do nothing
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
      Process::initialize();
      S0 = getVariableReference( "S0" );
      S1 = getVariableReference( "S1" );
      P0 = getVariableReference( "P0" );
      C0 = getVariableReference( "C0" );
    }

  virtual void process()
    {
      Real S0Concentration = S0.getConcentration();
      Real S1Concentration = S1.getConcentration();
      Real P0Concentration = P0.getConcentration();
      
      Real Den( KcF * KmP + KcF * P0Concentration 
		+ KcR * KmS0 * S1Concentration / Keq
		+ KcR * KmS1 * S0Concentration / Keq 
		+ KcF * P0Concentration * S1Concentration / KiS
		+ KcF * S0Concentration * S1Concentration / Keq );
      
      Real velocity = KcF * KcR * C0.getValue()
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

LIBECS_DM_INIT( OrderedBiUniProcess, Process );
