#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"
#include "PropertySlotMaker.hpp"
#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "VariableProxy.hpp"

#include "FluxProcess.hpp"
#include "ecell3_dm.hpp"

#define ECELL3_DM_TYPE Process
#define ECELL3_DM_CLASSNAME FM2Process

USE_LIBECS;

ECELL3_DM_CLASS
  :  
  public FluxProcess
{

  ECELL3_DM_OBJECT;
  
 public:

    ECELL3_DM_CLASSNAME()
    {
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, vm );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, Km );
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
	FluxProcess::initialize();
	P0 = getVariableReference( "P0" );
      }
    
 protected:
    
    Real vm;
    Real Km;
    VariableReference P0;
    
 private:
    
};

ECELL3_DM_INIT;
