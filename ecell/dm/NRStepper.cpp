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
  theTolerance( 0.0 ),
  theRng( gsl_rng_alloc( gsl_rng_mt19937 ) )
{
  // unset the default MinStepInterval.  
  setMinStepInterval( std::numeric_limits<Real>::min() );

  // set a seed.   
  // This can cause a problem in simultaneous multiple runs, because
  // time() can return the same value within 1 sec.
  gsl_rng_set( theRng, static_cast<unsigned long int>( time( NULL ) ) );

  CREATE_PROPERTYSLOT_GET    ( Real, TimeScale, NRStepper );
  CREATE_PROPERTYSLOT_SET_GET( Real, Tolerance, NRStepper );
}
	    
NRStepper::~NRStepper()
{
  gsl_rng_free( theRng );
}


void NRStepper::initialize()
{
  DiscreteEventStepper::initialize();

  // dynamic_cast theProcessVector of this Stepper to NRProcess, and store
  // it in theNRProcessVector.
  theNRProcessVector.clear();
  try
    {
      std::transform( theProcessVector.begin(), theProcessVector.end(),
		      std::back_inserter( theNRProcessVector ),
		      libecs::DynamicCaster<NRProcessPtr,ProcessPtr>() );
    }
  catch( const libecs::TypeError& )
    {
      THROW_EXCEPTION( InitializationFailed,
		       String( getClassName() ) + 
		       ": Only NRProcesses are allowed to exist "
		       "in this Stepper." );
    }

  // optimization: sort by memory address
  std::sort( theProcessVector.begin(), theProcessVector.end() );


  // (1) check Process dependency
  // (2) update step interval of each Process
  // (3) construct the priority queue (scheduler)
  thePriorityQueue.clear();
  const Real aCurrentTime( getCurrentTime() );
  for( NRProcessVector::const_iterator i( theNRProcessVector.begin() );
       i != theNRProcessVector.end(); ++i )
    {      
      NRProcessPtr anNRProcessPtr( *i );
	
      // check Process dependencies
      anNRProcessPtr->clearEffectList();
      // here assume aCoefficient != 0
      for( NRProcessVector::const_iterator j( theNRProcessVector.begin() );
	   j != theNRProcessVector.end(); ++j )
	{
	  NRProcessPtr const anNRProcess2Ptr( *j );
	  
	  if( anNRProcessPtr->checkEffect( anNRProcess2Ptr ) )
	    {
	      anNRProcessPtr->addEffect( anNRProcess2Ptr );
	    }
	}

      // warning: implementation dependent
      // here we assume size() is the index of the newly pushed element
      const Int anIndex( thePriorityQueue.size() );

      anNRProcessPtr->setIndex( anIndex );
      updateNRProcess( anNRProcessPtr );
      thePriorityQueue.push( NREvent( anNRProcessPtr->getStepInterval()
      				      + aCurrentTime,
      				      anNRProcessPtr ) );
    }

  // here all the NRProcesses are updated, then set new
  // step interval and reschedule this stepper.
  // That means, this Stepper doesn't necessary steps immediately
  // after initialize()
  NREventCref aTopEvent( thePriorityQueue.top() );
  const Real aNewTime( aTopEvent.getTime() );

  setStepInterval( aNewTime - getCurrentTime() );
  getModel()->reschedule( this );

}
  

// this doesn't necessary occur at the first step of the simulation,
// and imediately after initialize(), because initialize() recalculates
// all propensities and reschedules this stepper.
void NRStepper::step()
{
  NREventCref anEvent( thePriorityQueue.top() );

  NRProcessPtr const aMuProcess( anEvent.getProcess() );
  aMuProcess->process();

  // it assumes all coefficients are one or minus one
  // Process::initialize() should check this
  theTimeScale = 
    aMuProcess->getMinValue() * aMuProcess->getStepInterval() * theTolerance;

  const Real aCurrentTime( getCurrentTime() );
  // Update relevant mus
  NRProcessVectorCref anEffectList( aMuProcess->getEffectList() );
  for ( NRProcessVectorConstIterator i( anEffectList.begin() );
	i!= anEffectList.end(); ++i ) 
    {
      NRProcessPtr const anAffectedProcess( *i );
      updateNRProcess( anAffectedProcess );
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
  // update step intervals of NRProcesses
  const Real aCurrentTime( aCaller->getCurrentTime() );
  for( NRProcessVector::const_iterator i( theNRProcessVector.begin() );
       i != theNRProcessVector.end(); ++i )
    {      
      NRProcessPtr const anNRProcessPtr( *i );
	
      updateNRProcess( anNRProcessPtr );
      const Real aStepInterval( anNRProcessPtr->getStepInterval() );

      thePriorityQueue.changeOneKey( anNRProcessPtr->getIndex(),
				     NREvent( aStepInterval + aCurrentTime,
					      anNRProcessPtr ) );
    }

  NREventCref aTopEvent( thePriorityQueue.top() );
  const Real aNewTime( aTopEvent.getTime() );

  // reschedule this Stepper to aNewStepInterval past current time of
  // the interruption caller.
  setStepInterval( aNewTime - getCurrentTime() );
  getModel()->reschedule( this );
}


