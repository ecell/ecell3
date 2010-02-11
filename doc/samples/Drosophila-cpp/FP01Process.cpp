#include "libecs.hpp"
#include "ContinuousProcess.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( FP01Process, ContinuousProcess )
{

 public:

  LIBECS_DM_OBJECT( FP01Process, Process )
    {
      INHERIT_PROPERTIES( ContinuousProcess );

      PROPERTYSLOT_SET_GET( Real, Km );
    }

  FP01Process()
    {
      ; // do nothing
    }
  
  SIMPLE_SET_GET_METHOD( Real, Km );
  // expands 
  //void setKm( RealCref value ) { Km = value; }
  //const Real getKm() const { return Km; }

  virtual void fire()
    {
      Real E( C0.getMolarConc() );
      Real V( Km * E );
      V *= 1E-018 * N_A;
      setFlux( V );
    }

  virtual void initialize()
    {
      Process::initialize();
      C0 = getVariableReference( "C0" );
    }
    
 protected:
  Real Km;
  VariableReference C0;
  
 private:

};

LIBECS_DM_INIT( FP01Process, Process );
