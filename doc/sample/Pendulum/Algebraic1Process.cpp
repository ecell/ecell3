
#include "libecs.hpp"
#include "Process.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( Algebraic1Process, Process )
{

 public:

  LIBECS_DM_OBJECT( Algebraic1Process, Process )
    {
      INHERIT_PROPERTIES( Process );
    }

  Algebraic1Process()
    {
      ; // do nothing
    }

  virtual void initialize()
    {
      Process::initialize();

      C0 = getVariableReference( "C0" ); // X
      C1 = getVariableReference( "C1" ); // Y
      C2 = getVariableReference( "C2" ); // U
      C3 = getVariableReference( "C3" ); // V
    }

  virtual void fire()
    {
      setActivity( 2 * C0.getValue() * C2.getValue()
		   + 2 * C1.getValue() * C3.getValue() );
    }

 protected:
  
  VariableReference C0;
  VariableReference C1;
  VariableReference C2;
  VariableReference C3;
  
};

LIBECS_DM_INIT( Algebraic1Process, Process );
