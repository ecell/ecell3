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
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, V4 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, K4 );
    }
  
  SIMPLE_SET_GET_METHOD( Real, V4 );
  SIMPLE_SET_GET_METHOD( Real, K4 );
  // expands
  //void setV4( RealCref value ) { V4 = value; }
  //const Real getV4() const { return V4; }
  //void setK4( RealCref value ) { K4 = value; }
  //const Real getK4() const { return K4; }

  virtual void process()
    {
      Real E( C0.getConcentration() );
      
      Real V( -1 * V4 * E );
      V /= K4 + E;
      V *= 1E-018 * N_A;
      
      setFlux( V );
    }

  virtual void initialize()
    {
      FluxProcess::initialize();
      C0 = getVariableReference( "C0" );
    }

 protected:

  Real V4;
  Real K4;
  VariableReference C0;

 private:

};

ECELL3_DM_INIT;
