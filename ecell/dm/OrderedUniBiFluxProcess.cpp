#include "libecs.hpp"

#include "ContinuousProcess.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( OrderedUniBiFluxProcess, ContinuousProcess )
{

 public:

  LIBECS_DM_OBJECT( OrderedUniBiFluxProcess, Process )
    {
      INHERIT_PROPERTIES( ContinuousProcess );

      PROPERTYSLOT_SET_GET( Real, KcF );
      PROPERTYSLOT_SET_GET( Real, KcR );
      PROPERTYSLOT_SET_GET( Real, Keq );
      PROPERTYSLOT_SET_GET( Real, KmS );
      PROPERTYSLOT_SET_GET( Real, KmP0 );
      PROPERTYSLOT_SET_GET( Real, KmP1 );
      PROPERTYSLOT_SET_GET( Real, KiP );
    }
  

  OrderedUniBiFluxProcess()
    :
    KcF( 0.0 ),
    KcR( 0.0 ),
    Keq( 1.0 ),
    KmS( 1.0 ),
    KmP0( 1.0 ),
    KmP1( 1.0 ),
    KiP( 1.0 ),
    Keq_Inv( 1.0 )
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

      Keq_Inv = 1.0 / Keq;
    }

  virtual void fire()
    {
      Real S0Concentration = S0.getMolarConc();
      Real P0Concentration = P0.getMolarConc();
      Real P1Concentration = P1.getMolarConc();
      
      Real Den( KcR * KmS + KcR * S0Concentration 
		+ KcF * KmP1 * P0Concentration * Keq_Inv
		+ KcF * KmP0 * P1Concentration * Keq_Inv
		+ KcR * S0Concentration * P0Concentration / KiP
		+ KcR * P0Concentration * P1Concentration * Keq_Inv );
      Real velocity( KcF * KcR * C0.getValue()
		     * (S0Concentration - P0Concentration * 
			P1Concentration * Keq_Inv) / Den );
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

  Real Keq_Inv;

  VariableReference S0;
  VariableReference P0;
  VariableReference P1;
  VariableReference C0;
  
};

LIBECS_DM_INIT( OrderedUniBiFluxProcess, Process );
