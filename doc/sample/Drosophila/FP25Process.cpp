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
#define ECELL3_DM_CLASSNAME FP25Process

USE_LIBECS;

ECELL3_DM_CLASS
  :  
  public FluxProcess
{

  ECELL3_DM_OBJECT;

 public:

  ECELL3_DM_CLASSNAME()
    {
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, vd );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, Kd );
    }

  SIMPLE_SET_GET_METHOD( Real, vd );
  SIMPLE_SET_GET_METHOD( Real, Kd );
  // expands
  //void setvd( RealCref value ) { vd = value; }
  //const Real getvd() const { return vd; }
  //void setKd( RealCref value ) { Kd = value; }
  //const Real getKd() const { return Kd; }
    
  virtual void process()
    {
      Real E( C0.getConcentration() );
      
      Real V( -1 * vd * E );
      V /= Kd + E;
      V *= 1E-018 * N_A;

      setFlux( V );
    }

  virtual void initialize()
    {
      FluxProcess::initialize();
      C0 = getVariableReference( "C0" );
    }    
  
 protected:

  Real vd;
  Real Kd;
  VariableReference C0;

 private:

};

ECELL3_DM_INIT;
