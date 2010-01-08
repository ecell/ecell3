#include "libecs.hpp"
#include "ContinuousProcess.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( FP21Process, ContinuousProcess )
{

 public:

  LIBECS_DM_OBJECT( FP21Process, Process )
    {
      INHERIT_PROPERTIES( ContinuousProcess );

      PROPERTYSLOT_SET_GET( Real, V3 );
      PROPERTYSLOT_SET_GET( Real, K3 );
    }

  FP21Process()
    {
      ; // do nothing
    }

  SIMPLE_SET_GET_METHOD( Real, V3 );
  SIMPLE_SET_GET_METHOD( Real, K3 );
  // expands
  //void setV3( RealCref value ) { V3 = value; }
  //const Real getV3() const { return V3; }
  //void setK3( RealCref value ) { K3 = value; }
  //const Real getK3() const { return K3; }
    
  virtual void fire()
    {
      Real E( C0.getMolarConc() );
      
      Real V( V3 * E );
      V /= K3 + E;
      V *= 1E-018 * N_A;
      
      setFlux( V );
    }

  virtual void initialize()
    {
      Process::initialize();
      C0 = getVariableReference( "C0" );
    }

 protected:
  
  Real V3;
  Real K3;
  VariableReference C0;

 private:

};

LIBECS_DM_INIT( FP21Process, Process );
