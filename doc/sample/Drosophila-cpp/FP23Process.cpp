#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"
#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "VariableProxy.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( FP23Process, Process )
{

 public:

  LIBECS_DM_OBJECT( FP23Process, Process )
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( Real, k1 );
    }

  FP23Process()
    {
      ; // do nothing
    }

  SIMPLE_SET_GET_METHOD( Real, k1 );
  //void setk1( RealCref value ) { k1 = value; }
  //const Real getk1() const { return k1; }
    
  virtual void process()
    {
      Real E( C0.getMolarConc() );
      
      Real V( -1 * k1 * E );
      V *= 1E-018 * N_A;
      
      setFlux( V );
    }

  virtual void initialize()
    {
      Process::initialize();
      C0 = getVariableReference( "C0" );
    }


 protected:

  Real k1;
  VariableReference C0;

 private:

};

LIBECS_DM_INIT( FP23Process, Process );
