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
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, V2 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, K2 );
    }

  SIMPLE_SET_GET_METHOD( Real, V2 );
  SIMPLE_SET_GET_METHOD( Real, K2 );

  virtual void initialize()
    {
      C0 = getVariableReference( "C0" );
    }

  virtual void process()
    {
      Real E( C0.getConcentration() );

      Real V( -1 * V2 * E );
      V /= K2 + E;
      V *= 1E-018 * N_A;

      setFlux( V );
    }

  protected:
    
    Real V2;
    Real K2;

  VariableReference C0;
};

ECELL3_DM_INIT;

