
#include "libecs.hpp"
#include "ContinuousProcess.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( Differential1Process, ContinuousProcess )
{

 public:

  LIBECS_DM_OBJECT( Differential1Process, Process )
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( Real, k );
    }

  Differential1Process()
    {
      ; // do nothing
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

LIBECS_DM_INIT( Differential1Process, Process );
