
#include "libecs.hpp"
#include "Process.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( Algebraic3Process, Process )
{

 public:

  LIBECS_DM_OBJECT( Algebraic3Process, Process )
    {
      INHERIT_PROPERTIES( Process );
    }

  Algebraic3Process()
    {
      ; // do nothing
    }

  virtual void initialize()
    {
      Process::initialize();

      C0 = getVariableReference( "C0" );
      C1 = getVariableReference( "C1" );
    }

  virtual void fire()
    {
      setActivity( C0.getValue() * C0.getValue()
		   + C1.getValue() * C1.getValue() - 1.0 );
    }

 protected:
  
  VariableReference C0;
  VariableReference C1;
  
};

LIBECS_DM_INIT( Algebraic3Process, Process );
