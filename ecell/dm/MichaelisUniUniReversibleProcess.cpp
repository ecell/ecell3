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

LIBECS_DM_CLASS( MichaelisUniUniReversibleProcess, Process )
{

 public:

  LIBECS_DM_OBJECT( MichaelisUniUniReversibleProcess, Process )
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( Real, KmS );
      PROPERTYSLOT_SET_GET( Real, KmP );
      PROPERTYSLOT_SET_GET( Real, KcF );
      PROPERTYSLOT_SET_GET( Real, KcR );
    }
  


  // FIXME: property initial values?
  MichaelisUniUniReversibleProcess()
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

  virtual void process()
    {
      const Real S( S0.getConcentration() );
      const Real P( P0.getConcentration() );

      Real velocity( KcF * KmP * S );
      velocity -= KcR * KmS * P;
      velocity *= C0.getValue();
      
      Real Den( KmS * P );
      Den += KmP * S;
      Den += KmSP; 

      velocity /= Den;

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

LIBECS_DM_INIT( MichaelisUniUniReversibleProcess, Process );
