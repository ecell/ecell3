#include "libecs.hpp"
#include "ContinuousProcess.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( FP11Process, ContinuousProcess )
{

 public:

  LIBECS_DM_OBJECT( FP11Process, Process )
    {
      INHERIT_PROPERTIES( ContinuousProcess );

      PROPERTYSLOT_SET_GET( Real, V1 );
      PROPERTYSLOT_SET_GET( Real, K1 );
    }

  FP11Process()
    {
      ; // do nothing
    }

  SIMPLE_SET_GET_METHOD( Real, V1 );
  SIMPLE_SET_GET_METHOD( Real, K1 );
  // expands
  //void setV1( RealCref value ) { V1 = value; }
  //const Real getV1() const { return V1; }
  //void setK1( RealCref value ) { K1 = value; }
  //const Real getK1() const { return K1; }

  virtual void fire()
    {
      Real E( C0.getMolarConc() );
      Real V( V1 * E );
      V /= K1 + E;
      V *= 1E-018 * N_A;
      setFlux( V );
    }

  virtual void initialize()
    {
      Process::initialize();
      C0 = getVariableReference( "C0" );
    }

 protected:
  
  Real V1;
  Real K1;
  VariableReference C0;

 private:

};

LIBECS_DM_INIT( FP11Process, Process );
