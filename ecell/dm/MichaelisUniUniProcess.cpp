#ifndef __MichaelisUniUniProcess_CPP
#define __MichaelisUniUniProcess_CPP

#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"
#include "PropertySlotMaker.hpp"
#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "VariableProxy.hpp"

#include "FluxProcess.hpp"
#include "ecell3_dm.hpp"

#define ECELL3_DM_TYPE Process

USE_LIBECS;

ECELL3_DM_CLASS
    :  
  public FluxProcess
{

  ECELL3_DM_OBJECT;
  
 public:

  ECELL3_DM_CLASSNAME()
    {
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KmS );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KcF );
    }
  
  SIMPLE_SET_GET_METHOD( Real, KmS );
  SIMPLE_SET_GET_METHOD( Real, KcF );
    
  virtual void initialize()
    {
      FluxProcess::initialize();
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



ECELL3_DM_INIT;

#endif
