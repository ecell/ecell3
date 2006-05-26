#include "libecs.hpp"

#include "ContinuousProcess.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( OrderedBiUniFluxProcess, ContinuousProcess )
{

 public:

  LIBECS_DM_OBJECT( OrderedBiUniFluxProcess, Process )
    {
      INHERIT_PROPERTIES( ContinuousProcess );

      PROPERTYSLOT_SET_GET( Real, KcF );
      PROPERTYSLOT_SET_GET( Real, KcR );
      PROPERTYSLOT_SET_GET( Real, Keq );
      PROPERTYSLOT_SET_GET( Real, KmS0 );
      PROPERTYSLOT_SET_GET( Real, KmS1 );
      PROPERTYSLOT_SET_GET( Real, KmP );
      PROPERTYSLOT_SET_GET( Real, KiS );
    }


  OrderedBiUniFluxProcess()
    :
    KcF( 0.0 ),
    KcR( 0.0 ),
    Keq( 1.0 ),
    KmS0( 1.0 ),
    KmS1( 1.0 ),
    KmP( 1.0 ),
    KiS( 1.0 ),
    Keq_Inv( 1.0 )
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

      Keq_Inv = 1.0 / Keq;
    }

  virtual void fire()
    {
      Real S0Concentration = S0.getMolarConc();
      Real S1Concentration = S1.getMolarConc();
      Real P0Concentration = P0.getMolarConc();
      
      Real Den( KcF * KmP + KcF * P0Concentration 
		+ KcR * KmS0 * S1Concentration * Keq_Inv
		+ KcR * KmS1 * S0Concentration * Keq_Inv 
		+ KcF * P0Concentration * S1Concentration / KiS
		+ KcF * S0Concentration * S1Concentration * Keq_Inv );
      
      Real velocity = KcF * KcR * C0.getValue()
	* (P0Concentration  - S0Concentration  * S1Concentration * Keq_Inv )
	/ Den;    
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

  Real Keq_Inv;

  VariableReference S0;
  VariableReference S1;
  VariableReference P0;
  VariableReference C0;
  
};

LIBECS_DM_INIT( OrderedBiUniFluxProcess, Process );
