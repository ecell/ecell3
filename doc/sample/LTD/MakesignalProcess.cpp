#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"

#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "VariableProxy.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( MakesignalProcess, Process )
{
  
 public:
  
  LIBECS_DM_OBJECT( MakesignalProcess, Process )
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( Real, Impulse );
      PROPERTYSLOT_SET_GET( Real, Interval );
      PROPERTYSLOT_SET_GET( Real, Duration );
      PROPERTYSLOT_SET_GET( Real, DecayFactor );
    }
  

  MakesignalProcess()
    :
    Impulse( 0.0 ),
    Interval( 1.0 ),
    Duration( 0.0 ),
    DecayFactor( 10.0 )
    {
      ; // do nothing
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

  void fire()
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

LIBECS_DM_INIT( MakesignalProcess, Process );
