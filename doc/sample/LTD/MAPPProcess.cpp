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
  
  virtual void process()
    {
      const Real s0( S0.getValue() );
      const Real s1( S1.getValue() );
      const Real p( s0 + s1 );
      P0.setValue( p );    
    }
  
  virtual void initialize()
    {
      FluxProcess::initialize();
      S0 = getVariableReference( "S0" );
      S1 = getVariableReference( "S1" );
      P0 = getVariableReference( "P0" );
    }
  
 protected:
  VariableReference S0;
  VariableReference S1;
  VariableReference P0;
};

ECELL3_DM_INIT;
