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
#define ECELL3_DM_CLASSNAME FP21Process

USE_LIBECS;

ECELL3_DM_CLASS
  :  
  public FluxProcess
{

  ECELL3_DM_OBJECT;
  
 public:

  ECELL3_DM_CLASSNAME()
    {
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, V3 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, K3 );
    }

  SIMPLE_SET_GET_METHOD( Real, V3 );
  SIMPLE_SET_GET_METHOD( Real, K3 );
  // expands
  //void setV3( RealCref value ) { V3 = value; }
  //const Real getV3() const { return V3; }
  //void setK3( RealCref value ) { K3 = value; }
  //const Real getK3() const { return K3; }
    
  virtual void process()
    {
      Real E( C0.getConcentration() );
      
      Real V( V3 * E );
      V /= K3 + E;
      V *= 1E-018 * N_A;
      
      setFlux( V );
    }

  virtual void initialize()
    {
      FluxProcess::initialize();
      C0 = getVariableReference( "C0" );
    }

 protected:
  
  Real V3;
  Real K3;
  VariableReference C0;

 private:

};

ECELL3_DM_INIT;
