
#include "libecs.hpp"
#include "Process.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( Algebraic2Process, Process )
{

 public:

  LIBECS_DM_OBJECT( Algebraic2Process, Process )
    {
      INHERIT_PROPERTIES( Process );
    }

  Algebraic2Process()
    {
      ; // do nothing
    }

  virtual void initialize()
    {
      Process::initialize();

      C0 = getVariableReference( "C0" ); // Y
      C1 = getVariableReference( "C1" ); // U
      C2 = getVariableReference( "C2" ); // V
      C3 = getVariableReference( "C3" ); // T
    }

  virtual void fire()
    {
      setActivity( C1.getValue() * C1.getValue()
		   + C2.getValue() * C2.getValue()
		   - C3.getValue()
		   - C0.getValue() );
    }

 protected:
  
  VariableReference C0;
  VariableReference C1;
  VariableReference C2;
  VariableReference C3;

};

LIBECS_DM_INIT( Algebraic2Process, Process );
