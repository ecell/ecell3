#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"
#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "VariableProxy.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( FP03Process, Process )
{

 public:

  LIBECS_DM_OBJECT( FP03Process, Process )
    {
      INHERIT_PROPERTIES( Process );
      PROPERTYSLOT_SET_GET( Real, V2 );
      PROPERTYSLOT_SET_GET( Real, K2 );
    }

  FP03Process()
    {
      ; // do nothing
    }

  SIMPLE_SET_GET_METHOD( Real, V2 );
  SIMPLE_SET_GET_METHOD( Real, K2 );
  // expands
  //void setV2( RealCref value ) { V2 = value; }
  //const Real getV2() const { return V2; }
  //void setK2( RealCref value ) { K2 = value; }
  //const Real getK2() const { return K2; }

  virtual void process()
    {
      Real E( C0.getMolarConc() );
      
      Real V( V2 * E );
      V /= K2 + E;
      V *= 1E-018 * N_A;
      
      setFlux( V );
    }

  virtual void initialize()
    {
      Process::initialize();
      C0 = getVariableReference( "C0" );
    }

 protected:
  Real V2;
  Real K2;
  VariableReference C0;
 private:

};

LIBECS_DM_INIT( FP03Process, Process );
