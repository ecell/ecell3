#include "libecs.hpp"

#include "ContinuousProcess.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( FM1Process, ContinuousProcess )
{

 public:

  LIBECS_DM_OBJECT( FM1Process, Process )
    {
      INHERIT_PROPERTIES( ContinuousProcess );

      PROPERTYSLOT_SET_GET( Real, vs );
      PROPERTYSLOT_SET_GET( Real, KI );
    }

  FM1Process()
     {
       ; // do nothing
     }

   SIMPLE_SET_GET_METHOD( Real, vs );
   SIMPLE_SET_GET_METHOD( Real, KI );
   //void setvs( RealCref value ) { vs = value; }
   //const Real getvs() const { return vs; }
   //void setKI( RealCref value ) { KI = value; }
   //const Real getKI() const { return KI; }

    virtual void initialize()
      {
	Process::initialize();
	C0 = getVariableReference( "C0" );
      }

    virtual void fire()
      {
	Real E( C0.getMolarConc() );
	Real V( vs * KI );
	V /= KI + (E * E * E);
	V *= 1E-018 * N_A;
	setFlux( V );
      }

 protected:

    Real vs;
    Real KI;

    VariableReference C0;

  private:

};

LIBECS_DM_INIT( FM1Process, Process );

