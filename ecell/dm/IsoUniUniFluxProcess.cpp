#include "libecs.hpp"
#include "Util.hpp"

#include "ContinuousProcess.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( IsoUniUniFluxProcess, ContinuousProcess )
{

  
 public:

  LIBECS_DM_OBJECT( IsoUniUniFluxProcess, Process )
    {
      INHERIT_PROPERTIES( ContinuousProcess );

      PROPERTYSLOT_SET_GET( Real, KmS );
      PROPERTYSLOT_SET_GET( Real, KmP );
      PROPERTYSLOT_SET_GET( Real, KcF );
      PROPERTYSLOT_SET_GET( Real, Keq );
      PROPERTYSLOT_SET_GET( Real, KiiP );
    }

  IsoUniUniFluxProcess()
    :
    KmS( 1.0 ),
    KmP( 1.0 ),
    KcF( 0.0 ),
    Keq( 0.0 ),
    KiiP( 1.0 )
    {
      ; // do nothing
    }
  
  SIMPLE_SET_GET_METHOD( Real, KmS );
  SIMPLE_SET_GET_METHOD( Real, KmP );
  SIMPLE_SET_GET_METHOD( Real, KcF );
  SIMPLE_SET_GET_METHOD( Real, Keq );
  SIMPLE_SET_GET_METHOD( Real, KiiP );
    
  virtual void initialize()
    {
      Process::initialize();
      S0 = getVariableReference( "S0" );
      P0 = getVariableReference( "P0" );
      C0 = getVariableReference( "C0" );  
    }

  virtual void process()
    {
      Real S( S0.getMolarConc() );
      Real P( P0.getMolarConc() );
      Real velocity( KcF * C0.getMolarConc() * 
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

LIBECS_DM_INIT( IsoUniUniFluxProcess, Process );
