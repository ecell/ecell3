#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"
#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "VariableProxy.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( FP14Process, Process )
{

 public:

  LIBECS_DM_OBJECT( FP14Process, Process )
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( Real, V4 );
      PROPERTYSLOT_SET_GET( Real, K4 );
    }

  FP14Process()
    {
      ; // do nothing
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
      
      Real V( V4 * E );
      V /= K4 + E;
      V *= 1E-018 * N_A;

      setFlux( V );
    }

  virtual void initialize()
    {
      Process::initialize();
      C0 = getVariableReference( "C0" );
    }

 protected:

  Real V4;
  Real K4;
  VariableReference C0;

 private:

};

LIBECS_DM_INIT( FP14Process, Process );
