
#include "libecs.hpp"
#include "ContinuousProcess.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( Differential2Process, ContinuousProcess )
{

 public:

  LIBECS_DM_OBJECT( Differential2Process, Process )
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( Real, c );
    }

  Differential2Process()
    {
      ; // do nothing
    }

  SIMPLE_SET_GET_METHOD( Real, c );
    
  virtual void initialize()
    {
      Process::initialize();

      C0 = getVariableReference( "C0" );
      C1 = getVariableReference( "C1" );
    }

  virtual void fire()
    {
      Real aVelocity( c );

      aVelocity -= C0.getValue() * C1.getValue();

      setFlux( aVelocity );
    }

  virtual const bool isContinuous() const
    {
      return true;
    }

 protected:
  
  Real c;
  VariableReference C0;
  VariableReference C1;
  
};

LIBECS_DM_INIT( Differential2Process, Process );
