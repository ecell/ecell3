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

LIBECS_DM_CLASS( MichaelisUniUniProcess, Process )
{

 public:

  LIBECS_DM_OBJECT( MichaelisUniUniProcess, Process )
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( Real, KmS );
      PROPERTYSLOT_SET_GET( Real, KcF );
    }  

  //FIXME: property initial values?
  MichaelisUniUniProcess()
    {
      ; // do nothing
    }
  
  SIMPLE_SET_GET_METHOD( Real, KmS );
  SIMPLE_SET_GET_METHOD( Real, KcF );
    
  virtual void initialize()
    {
      Process::initialize();
      S0 = getVariableReference( "S0" );
      C0 = getVariableReference( "C0" );  
    }

  virtual void process()
    {
      Real velocity( KcF );
      velocity *= C0.getValue();
      const Real S( S0.getConcentration() );
      velocity *= S;
      velocity /= ( KmS + S );
      setFlux( velocity );
    }

 protected:
  
  Real KmS;
  Real KcF;
  VariableReference S0;
  VariableReference C0;
  
};

LIBECS_DM_INIT( MichaelisUniUniProcess, Process );
