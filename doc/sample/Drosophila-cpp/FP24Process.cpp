#include "libecs.hpp"
#include "ContinuousProcess.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( FP24Process, ContinuousProcess )
{

 public:

  LIBECS_DM_OBJECT( FP24Process, Process )
    {
      INHERIT_PROPERTIES( ContinuousProcess );

      PROPERTYSLOT_SET_GET( Real, k2 );
    }

  FP24Process()
    {
      ; // do nothing
    }

  SIMPLE_SET_GET_METHOD( Real, k2 );
  // expands
  //void setk2( RealCref value ) { k2 = value; }
  //const Real getk2() const { return k2; }
    
  virtual void fire()
    {
      Real E( C0.getMolarConc() );
      
      Real V( k2 * E );
      V *= 1E-018 * N_A;
      
      setFlux( V );
    }

  virtual void initialize()
    {
      Process::initialize();
      C0 = getVariableReference( "C0" );
    }    

 protected:

  Real k2;
  VariableReference C0;

 private:

};

LIBECS_DM_INIT( FP24Process, Process );
