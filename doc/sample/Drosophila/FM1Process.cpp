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
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, vs );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KI );
    }

  SIMPLE_SET_GET_METHOD( Real, vs );
  SIMPLE_SET_GET_METHOD( Real, KI );

  virtual void initialize()
    {
      FluxProcess::initialize();
      C0 = getVariableReference( "C0" );
    }

  virtual void process()
    {
      Real E( C0.getConcentration() );
      Real V( vs * KI );
      V /= KI + (E * E * E);
      V *= 1E-018 * N_A;
      setFlux( V );
    }

 protected:

  Real vs;
  Real KI;

  VariableReference C0;

};

ECELL3_DM_INIT;
