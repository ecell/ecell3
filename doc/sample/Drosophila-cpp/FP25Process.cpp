#include "libecs.hpp"
#include "ContinuousProcess.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( FP25Process, ContinuousProcess )
{

 public:

  LIBECS_DM_OBJECT( FP25Process, Process )
    {
      INHERIT_PROPERTIES( ContinuousProcess );

      PROPERTYSLOT_SET_GET( Real, vd );
      PROPERTYSLOT_SET_GET( Real, Kd );
    }

  FP25Process()
    {
      ; // do nothing
    }


  SIMPLE_SET_GET_METHOD( Real, vd );
  SIMPLE_SET_GET_METHOD( Real, Kd );
  // expands
  //void setvd( RealCref value ) { vd = value; }
  //const Real getvd() const { return vd; }
  //void setKd( RealCref value ) { Kd = value; }
  //const Real getKd() const { return Kd; }
    
  virtual void fire()
    {
      Real E( C0.getMolarConc() );
      
      Real V( -1 * vd * E );
      V /= Kd + E;
      V *= 1E-018 * N_A;

      setFlux( V );
    }

  virtual void initialize()
    {
      Process::initialize();
      C0 = getVariableReference( "C0" );
    }    
  
 protected:

  Real vd;
  Real Kd;
  VariableReference C0;

 private:

};

LIBECS_DM_INIT( FP25Process, Process );
