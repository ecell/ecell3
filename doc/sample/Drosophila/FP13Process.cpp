#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"
#include "PropertySlotMaker.hpp"
#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "VariableProxy.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( FP13Process, Process )
{

 public:

  LIBECS_DM_OBJECT( FP13Process, Process )
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( Real, V3 );
      PROPERTYSLOT_SET_GET( Real, K3 );
    }

  FP13Process()
    {
      ; // do nothing
    }

  SIMPLE_SET_GET_METHOD( Real, V3 );
  SIMPLE_SET_GET_METHOD( Real, K3 );
  // expands
  //void setV3( RealCref value ) { V3 = value; }
  //const Real getV3() const { return V3; }
  //void setK3( RealCref value ) { K3 = value; }
  //const Real getK3() const { return K3; }
    

  virtual void initialize()
    {
      Process::initialize();
      C0 = getVariableReference( "C0" );
    }
  
  virtual void process()
    {
      Real E( C0.getConcentration() );
      
      Real V( -1 * V3 * E );
      V /= K3 + E;
      V *= 1E-018 * N_A;
      
      setFlux( V );
    }

 protected:

  Real V3;
  Real K3;
  VariableReference C0;

 private:
  
};

LIBECS_DM_INIT( FP13Process, Process );
