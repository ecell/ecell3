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

#include "NRStepper.hpp"

#include "NRProcess.hpp"



DM_INIT( Stepper, NRStepper );



NRStepper::NRStepper()
  :
  theTimeScale( 0.0 ),
  theTolerance( 0.1 ),
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

  theNRProcessVector.clear();
  for( ProcessVector::const_iterator i( theProcessVector.begin() );
       i != theProcessVector.end(); ++i )
    {
      ProcessPtr aProcessPtr( *i );

      NRProcessPtr 
	anNRProcessPtr( dynamic_cast<NRProcessPtr>( aProcessPtr ) );
      if( anNRProcessPtr != NULLPTR )
	{
	  theNRProcessVector.push_back( anNRProcessPtr );
	}
      else
	{
	  THROW_EXCEPTION( InitializationFailed,
			   String( getClassName() ) + 
			   ": Only NRProcesses are allowed to exist "
			   "in this Stepper." );
	}
    }


  const Real aCurrentTime( getCurrentTime() );
  thePriorityQueue.clear();
  for( NRProcessVector::const_iterator i( theNRProcessVector.begin() );
       i != theNRProcessVector.end(); ++i )
    {      
      NRProcessPtr anNRProcessPtr( *i );
	
      // warning: implementation dependent
      // here we assume size() is the index of the newly pushed element
      const Int anIndex( thePriorityQueue.size() );

      anNRProcessPtr->setIndex( anIndex );
      anNRProcessPtr->updateStepInterval();
	
      thePriorityQueue.push( NREvent( anNRProcessPtr->getStepInterval()
				      + aCurrentTime,
				      anNRProcessPtr ) );
    }

}
  

void NRStepper::step()
{
  NREventCref anEvent( thePriorityQueue.top() );

  NRProcessPtr const aMuProcess( anEvent.getProcess() );
  aMuProcess->process();

  // it assumes all coefficients are one
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
      anAffectedProcess->updateStepInterval();
      const Real aStepInterval( anAffectedProcess->getStepInterval() );
      // aTime is time in the priority queue

      Int anIndex( anAffectedProcess->getIndex() );
      thePriorityQueue.changeOneKey( anIndex,
				     NREvent( aStepInterval + aCurrentTime,
					      anAffectedProcess ) );
      //	  std::cerr << "change: " << anIndex << ' ' << aStepInterval << ' ' << anAffectedProcess->getID() << std::endl;

    }

  NREventCref aTopEvent( thePriorityQueue.top() );
  const Real aNextStepInterval( aTopEvent.getTime() - aCurrentTime );

  setStepInterval( aNextStepInterval );

  //      std::cerr << aNextStepInterval << std::endl;

  //NRProcessPtr const aNextProcess( aTopEvent.getProcess() );
  //std::cerr << "next: " << aNextProcess->getID() << std::endl;
  //      std::cerr << "next: " << aNextProcess->getStepInterval() << std::endl;
  //      std::cerr << "next: " << aNextProcess->getMu() << std::endl;

}


void NRStepper::interrupt( StepperPtr const aCaller )
{
  // update step intervals of NRProcesses
  const Real aCurrentTime( aCaller->getCurrentTime() );
  for( NRProcessVector::const_iterator i( theNRProcessVector.begin() );
       i != theNRProcessVector.end(); ++i )
    {      
      NRProcessPtr const anNRProcessPtr( *i );
	
      anNRProcessPtr->updateStepInterval();
      const Real aStepInterval( anNRProcessPtr->getStepInterval() );

      thePriorityQueue.changeOneKey( anNRProcessPtr->getIndex(),
				     NREvent( aStepInterval + aCurrentTime,
					      anNRProcessPtr ) );
    }

  NREventCref aTopEvent( thePriorityQueue.top() );
  const Real aCallerCurrentTime( aCaller->getCurrentTime() );
  const Real aNewTime( aTopEvent.getTime() );

  // reschedule this Stepper to aNewStepInterval past current time of
  // the interruption caller.
  setStepInterval( aNewTime - aCallerCurrentTime );
  getModel()->reschedule( this );
}


