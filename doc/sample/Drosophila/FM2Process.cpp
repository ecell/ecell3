#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"
#include "PropertySlotMaker.hpp"
#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "VariableProxy.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( FM2Process, Process )
{

 public:

  LIBECS_DM_OBJECT( FM2Process, Process )
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( Real, vm );
      PROPERTYSLOT_SET_GET( Real, Km );
    }

  FM2Process()
     {
       ; // do nothing
     }

    SIMPLE_SET_GET_METHOD( Real, vm );
    SIMPLE_SET_GET_METHOD( Real, Km );
    //void setvm( RealCref value ) { vm = value; }
    //const Real getvm() const { return vm; }
    //void setKm( RealCref value ) { Km = value; }
    //const Real getKm() const { return Km; }
    
    virtual void process()
      {
	Real E( P0.getConcentration() );
	
	Real V( -1 * vm * E );
	V /= Km + E;
	V *= 1E-018 * N_A;
	
	setFlux( V );
      }
    
    virtual void initialize()
      {
	Process::initialize();
	P0 = getVariableReference( "P0" );
      }
    
 protected:
    
    Real vm;
    Real Km;
    VariableReference P0;
    
 private:
    
};

LIBECS_DM_INIT( FM2Process, Process );
