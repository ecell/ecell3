#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"
#include "PropertySlotMaker.hpp"
#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "VariableProxy.hpp"

#include "ecell3_dm.hpp"

#define ECELL3_DM_TYPE Process

USE_LIBECS;

ECELL3_DM_CLASS
  :  
  public Process
{
  
  ECELL3_DM_OBJECT;
  
 public:
  
  ECELL3_DM_CLASSNAME()
    :
    Impulse( 0.0 ),
    Interval( 1.0 ),
    Duration( 0.0 ),
    DecayFactor( 10.0 )
    {
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, Impulse );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, Interval );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, Duration );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, DecayFactor );
    }
  
  SIMPLE_SET_GET_METHOD( Real, Impulse );
  SIMPLE_SET_GET_METHOD( Real, Interval );
  SIMPLE_SET_GET_METHOD( Real, Duration );
  SIMPLE_SET_GET_METHOD( Real, DecayFactor );
  
  void initialize()
    {
      Process::initialize();
      P0 = getVariableReference( "P0" );
    }

  void process()
  {
    if( theLastTime < 0.0 )
      {
	return;
      }

    Real aCurrentTime( getStepper()->getCurrentTime() );

    Real aNextTime( theLastTime + Interval );
    
    if( aNextTime >= Duration )
      {
	theLastTime = -1;

	P0.setValue( 1e-20 * getSuperSystem()->getSizeN_A() );
	return;
      }					


    Real aTimeDifference( aCurrentTime - theLastTime );
    Real aDecay( aTimeDifference * DecayFactor / Interval );
    
    P0.setValue( Impulse * getSuperSystem()->getSizeN_A() 
		 / ( aDecay + 1 ) );
    
    if( aTimeDifference >= Interval )
      {
	theLastTime += Interval;
      }
  }
  
 protected:

  Real theLastTime;

  Real Impulse;
  Real Interval;
  Real Duration;
  Real DecayFactor;
  
  VariableReference P0;  
  
};

ECELL3_DM_INIT;
