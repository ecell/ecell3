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
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, k2 );
    }
  
  SIMPLE_SET_GET_METHOD( Real, k2 );

  virtual void process()
  {
    Real E( C0.getConcentration() );
    
    Real V( k2 * E );
    V *= 1E-018 * N_A;
    
    setFlux( V );
  }
  
  virtual void initialize()
    {
      FluxProcess::initialize();
      C0 = getVariableReference( "C0" );
    }
  
 protected:
  
  Real k2;
  VariableReference C0;

};

ECELL3_DM_INIT;