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
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, V1 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, K1 );
    }

  SIMPLE_SET_GET_METHOD( Real, V1 );
  SIMPLE_SET_GET_METHOD( Real, K1 );
  // expands
  //void setV1( RealCref value ) { V1 = value; }
  //const Real getV1() const { return V1; }
  //void setK1( RealCref value ) { K1 = value; }
  //const Real getK1() const { return K1; }

  virtual void process()
    {
      Real E( C0.getConcentration() );
      Real V( V1 * E );
      V /= K1 + E;
      V *= 1E-018 * N_A;
      setFlux( V );
    }

  virtual void initialize()
    {
      FluxProcess::initialize();
      C0 = getVariableReference( "C0" );
    }

 protected:
  
  Real V1;
  Real K1;
  VariableReference C0;

 private:

};

ECELL3_DM_INIT;
