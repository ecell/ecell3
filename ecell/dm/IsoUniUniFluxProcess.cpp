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

LIBECS_DM_CLASS( IsoUniUniProcess, Process )
{

  
 public:

  LIBECS_DM_OBJECT( IsoUniUniProcess, Process )
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( Real, KmS );
      PROPERTYSLOT_SET_GET( Real, KmP );
      PROPERTYSLOT_SET_GET( Real, KcF );
      PROPERTYSLOT_SET_GET( Real, Keq );
      PROPERTYSLOT_SET_GET( Real, KiiP );
    }

  //FIXME: initial values
  IsoUniUniProcess()
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
      Real S( S0.getConcentration() );
      Real P( P0.getConcentration() );
      Real velocity( KcF * C0.getConcentration() * 
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

LIBECS_DM_INIT( IsoUniUniProcess, Process );
