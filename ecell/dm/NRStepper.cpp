//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 2002 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-CELL is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-CELL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-CELL -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Hu Bin <hubin@sfc.keio.ac.jp> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//
//
// written by Kouichi Takahashi <shafi@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//


#include <time.h>

#include <limits>

#include <libecs/Util.hpp>

#include "NRStepper.hpp"



DM_INIT( Stepper, NRStepper );


NRStepper::NRStepper()
  :
  theTimeScale( 0.0 ),
  theTolerance( 0.0 )
{
  // unset the default Min/MaxStepInterval.  
  setMinStepInterval( std::numeric_limits<Real>::min() );
  setMaxStepInterval( std::numeric_limits<Real>::max() );

  CREATE_PROPERTYSLOT_GET    ( Real, TimeScale, NRStepper );
  CREATE_PROPERTYSLOT_SET_GET( Real, Tolerance, NRStepper );
}
	    
NRStepper::~NRStepper()
{
  ; // do nothing
}


void NRStepper::initialize()
{
  DiscreteEventStepper::initialize();

  // dynamic_cast theProcessVector of this Stepper to GillespieProcess, and store
  // it in theGillespieProcessVector.
  theGillespieProcessVector.clear();
  try
    {
      std::transform( theProcessVector.begin(), theProcessVector.end(),
		      std::back_inserter( theGillespieProcessVector ),
		      libecs::DynamicCaster<GillespieProcessPtr,ProcessPtr>() );
    }
  catch( const libecs::TypeError& )
    {
      THROW_EXCEPTION( InitializationFailed,
		       String( getClassName() ) + 
		       ": Only GillespieProcesses are allowed to exist "
		       "in this Stepper." );
    }

  // optimization: sort by memory address
  std::sort( theProcessVector.begin(), theProcessVector.end() );


  // (1) check Process dependency
  // (2) update step interval of each Process
  // (3) construct the priority queue (scheduler)
  thePriorityQueue.clear();
  const Real aCurrentTime( getCurrentTime() );
  for( GillespieProcessVector::const_iterator i( theGillespieProcessVector.begin() );
       i != theGillespieProcessVector.end(); ++i )
    {      
      GillespieProcessPtr anGillespieProcessPtr( *i );
	
      // check Process dependencies
      anGillespieProcessPtr->clearEffectList();
      // here assume aCoefficient != 0
      for( GillespieProcessVector::const_iterator j( theGillespieProcessVector.begin() );
	   j != theGillespieProcessVector.end(); ++j )
	{
	  GillespieProcessPtr const anGillespieProcess2Ptr( *j );
	  
	  if( anGillespieProcessPtr->checkEffect( anGillespieProcess2Ptr ) )
	    {
	      anGillespieProcessPtr->addEffect( anGillespieProcess2Ptr );
	    }
	}

      // warning: implementation dependent
      // here we assume size() is the index of the newly pushed element
      const Int anIndex( thePriorityQueue.size() );

      anGillespieProcessPtr->setIndex( anIndex );
      updateGillespieProcess( anGillespieProcessPtr );
      thePriorityQueue.push( NREvent( anGillespieProcessPtr->getStepInterval()
      				      + aCurrentTime,
      				      anGillespieProcessPtr ) );
    }

  // here all the GillespieProcesses are updated, then set new
  // step interval and reschedule this stepper.
  // That means, this Stepper doesn't necessary steps immediately
  // after initialize()
  NREventCref aTopEvent( thePriorityQueue.top() );
  const Real aNewTime( aTopEvent.getTime() );

  setStepInterval( aNewTime - aCurrentTime );
  getModel()->reschedule( this );

}
  

// this doesn't necessary occur at the first step of the simulation,
// and imediately after initialize(), because initialize() recalculates
// all propensities and reschedules this stepper.
void NRStepper::step()
{
  NREventCref anEvent( thePriorityQueue.top() );

  GillespieProcessPtr const aMuProcess( anEvent.getProcess() );
  aMuProcess->process();

  // it assumes all coefficients are one or minus one
  // Process::initialize() should check this
  theTimeScale = 
    aMuProcess->getMinValue() * aMuProcess->getStepInterval() * theTolerance;

  const Real aCurrentTime( getCurrentTime() );
  // Update relevant mus
  GillespieProcessVectorCref anEffectList( aMuProcess->getEffectList() );
  for ( GillespieProcessVectorConstIterator i( anEffectList.begin() );
	i!= anEffectList.end(); ++i ) 
    {
      GillespieProcessPtr const anAffectedProcess( *i );
      updateGillespieProcess( anAffectedProcess );
      const Real aStepInterval( anAffectedProcess->getStepInterval() );
      // aTime is time in the priority queue

      Int anIndex( anAffectedProcess->getIndex() );
      thePriorityQueue.changeOneKey( anIndex,
				     NREvent( aStepInterval + aCurrentTime,
					      anAffectedProcess ) );
    }

  NREventCref aTopEvent( thePriorityQueue.top() );
  const Real aNextStepInterval( aTopEvent.getTime() - aCurrentTime );

  setStepInterval( aNextStepInterval );
}


void NRStepper::interrupt( StepperPtr const aCaller )
{
  // update step intervals of GillespieProcesses
  const Real aCurrentTime( aCaller->getCurrentTime() );
  for( GillespieProcessVector::const_iterator i( theGillespieProcessVector.begin() );
       i != theGillespieProcessVector.end(); ++i )
    {      
      GillespieProcessPtr const anGillespieProcessPtr( *i );
	
      updateGillespieProcess( anGillespieProcessPtr );
      const Real aStepInterval( anGillespieProcessPtr->getStepInterval() );

      thePriorityQueue.changeOneKey( anGillespieProcessPtr->getIndex(),
				     NREvent( aStepInterval + aCurrentTime,
					      anGillespieProcessPtr ) );
    }

  NREventCref aTopEvent( thePriorityQueue.top() );
  const Real aNewTime( aTopEvent.getTime() );

  // reschedule this Stepper to aNewStepInterval past current time of
  // the interruption caller.
  setStepInterval( aNewTime - getCurrentTime() );
  getModel()->reschedule( this );
}


