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
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, vm );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, Km );
    }

  SIMPLE_SET_GET_METHOD( Real, vm );
  SIMPLE_SET_GET_METHOD( Real, Km );
  
  virtual void itialize()
    {
      FluxProcess::initialize();
      P0 = getVariableReference( "P0" );
    }

  virtual void process()
    {
      Real E( P0.getConcentration() );
      Real V( -1 * vm * E );
      V /= Km + E;
      V *= 1E-018 * N_A;

      setFlux( V );
    }

 protected:


  Real vm;
  Real Km;

  VariableReference P0;

};

ECELL3_DM_INIT;
