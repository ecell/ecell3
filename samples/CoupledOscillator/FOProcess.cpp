#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"

#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "Interpolant.hpp"

#include "Process.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( FOProcess, Process )
{

 public:

  LIBECS_DM_OBJECT( FOProcess, Process )
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( Real, k );
    }
  
  FOProcess()
    {
      ;
    }

  SIMPLE_SET_GET_METHOD( Real, k );
    
  virtual void initialize()
    {
      Process::initialize();
      C0 = getVariableReference( "C0" );  
    }

  virtual void fire()
    {
      Real aVelocity( k );

      aVelocity *= C0.getValue();

      setFlux( aVelocity );
    }

 protected:
  
  Real k;

  VariableReference C0;
  
};

LIBECS_DM_INIT( FOProcess, Process );
