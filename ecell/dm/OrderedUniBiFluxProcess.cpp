#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"

#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "VariableProxy.hpp"

#include "Process.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( OrderedUniBiProcess, Process )
{

 public:

  LIBECS_DM_OBJECT( OrderedUniBiProcess, Process )
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( Real, KcF );
      PROPERTYSLOT_SET_GET( Real, KcR );
      PROPERTYSLOT_SET_GET( Real, Keq );
      PROPERTYSLOT_SET_GET( Real, KmS );
      PROPERTYSLOT_SET_GET( Real, KmP0 );
      PROPERTYSLOT_SET_GET( Real, KmP1 );
      PROPERTYSLOT_SET_GET( Real, KiP );
    }
  

  // FIXME: initial values
  OrderedUniBiProcess()
    {
      ; // do nothing
    }
  
  SIMPLE_SET_GET_METHOD( Real, KcF );
  SIMPLE_SET_GET_METHOD( Real, KcR );
  SIMPLE_SET_GET_METHOD( Real, Keq );
  SIMPLE_SET_GET_METHOD( Real, KmS );
  SIMPLE_SET_GET_METHOD( Real, KmP0 );
  SIMPLE_SET_GET_METHOD( Real, KmP1 );
  SIMPLE_SET_GET_METHOD( Real, KiP );
    
  virtual void initialize()
    {
      Process::initialize();
      S0 = getVariableReference( "S0" );
      P0 = getVariableReference( "P0" );
      P1 = getVariableReference( "P1" );
      C0 = getVariableReference( "C0" );
    }

  virtual void process()
    {
      Real S0Concentration = S0.getConcentration();
      Real P0Concentration = P0.getConcentration();
      Real P1Concentration = P1.getConcentration();
      
      Real Den( KcR * KmS + KcR * S0Concentration 
		+ KcF * KmP1 * P0Concentration / Keq 
		+ KcF * KmP0 * P1Concentration / Keq 
		+ KcR * S0Concentration * P0Concentration / KiP
		+ KcR * P0Concentration * P1Concentration / Keq );
      Real velocity( KcF * KcR * C0.getValue()
		     * (S0Concentration - P0Concentration * P1Concentration / Keq) / Den );
      setFlux( velocity );
    }

 protected:

  Real KcF;
  Real KcR;
  Real Keq;

  Real KmS;
  Real KmP0;
  Real KmP1;
  Real KiP;

  VariableReference S0;
  VariableReference P0;
  VariableReference P1;
  VariableReference C0;
  
};

LIBECS_DM_INIT( OrderedUniBiProcess, Process );
