#include "libecs.hpp"

#include "ContinuousProcess.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( MichaelisUniUniFluxProcess, ContinuousProcess )
{

 public:

  LIBECS_DM_OBJECT( MichaelisUniUniFluxProcess, Process )
    {
      INHERIT_PROPERTIES( ContinuousProcess );

      PROPERTYSLOT_SET_GET( Real, KmS );
      PROPERTYSLOT_SET_GET( Real, KmP );
      PROPERTYSLOT_SET_GET( Real, KcF );
      PROPERTYSLOT_SET_GET( Real, KcR );
    }
  


  MichaelisUniUniFluxProcess()
    :
    KmS( 1.0 ),
    KmP( 1.0 ),
    KcF( 0.0 ),
    KcR( 0.0 ),
    KmSP( 1.0 )
    {
      // do nothing
    }
  
  SIMPLE_SET_GET_METHOD( Real, KmS );
  SIMPLE_SET_GET_METHOD( Real, KmP );
  SIMPLE_SET_GET_METHOD( Real, KcF );
  SIMPLE_SET_GET_METHOD( Real, KcR );
    
  virtual void initialize()
    {
      Process::initialize();
      
      KmSP = KmS * KmP;

      S0 = getVariableReference( "S0" );
      P0 = getVariableReference( "P0" );
      C0 = getVariableReference( "C0" );  
    }

  virtual void fire()
    {
      const Real S( S0.getMolarConc() );
      const Real P( P0.getMolarConc() );

      const Real KmP_S( KmP * S );
      const Real KmS_P( KmS * P );

      Real velocity( C0.getValue() * KcF * KmP_S );
      velocity -= KcR * KmS_P;
      
      velocity /= KmS_P + KmP_S + KmSP;

      setFlux( velocity );
    }

 protected:
  

  Real KmS;
  Real KmP;
  Real KcF;
  Real KcR;
  
  Real KmSP;

  VariableReference S0;
  VariableReference P0;
  VariableReference C0;
  
};

LIBECS_DM_INIT( MichaelisUniUniFluxProcess, Process );
