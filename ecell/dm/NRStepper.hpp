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


/***************************************************************************

 This work is based on Michael A. Gibson and Jehoshua Bruck 's Next
 Reaction Theory

 Michael A. Gibson and Jehoshua Bruck 

 " Efficient Exact Stochastic Simulation of Chemical Systems with Many
 Species and Many Channels " The Journal of Physical Chemistry A,
 104,9,1876-89 (2000).

***************************************************************************/


#ifndef __NRSTEPPER_HPP
#define __NRSTEPPER_HPP

//#include <iostream>
//#include <vector>
#include <algorithm>

#include <gsl/gsl_rng.h>

#include "libecs/DynamicPriorityQueue.hpp"
#include "libecs/DiscreteEventStepper.hpp"

#include "GillespieProcess.hpp"

USE_LIBECS;

DECLARE_CLASS( NRStepper );

class NRStepper 
  : 
  public DiscreteEventStepper 
{

  LIBECS_DM_OBJECT( Stepper, NRStepper );

  DECLARE_CLASS( NREvent );
  DECLARE_TYPE( DynamicPriorityQueue<NREvent>, NRPriorityQueue );


protected:

  // A pair of (reaction index, time) for inclusion in the priority queue.
  class NREvent
  {
  public:

    NREvent()
    {
      ; // do nothing
    }

    NREvent( RealCref aTime, GillespieProcessPtr aProcess )
      :
      theTime( aTime ),
      theProcess( aProcess )
    {
      ; // do nothing
    }

    const Real getTime() const
    {
      return theTime;
    }

    GillespieProcessPtr getProcess() const
    {
      return theProcess;
    }

    const bool operator< ( NREventCref rhs ) const
    {
      return theTime < rhs.theTime;
    }

    const bool operator!= ( NREventCref rhs ) const
    {
      return theTime != rhs.theTime || 
	theProcess != rhs.theProcess;
    }


  private:

    Real       theTime;
    GillespieProcessPtr theProcess;


  };

public:

  NRStepper(void);

  virtual ~NRStepper(void);

  virtual void initialize();

  virtual void step();

  virtual void interrupt( StepperPtr const aCaller );


  SET_METHOD( Real, Tolerance )
  {
    theTolerance = value;
  }

  GET_METHOD( Real, Tolerance )
  {
    return theTolerance;
  }

  virtual GET_METHOD( Real, TimeScale )
  {
    //return theLastProcess->getTimeScale();
    return theTimeScale;
  }

  const gsl_rng* getRng() const
  {
    return theRng;
  }
    
  void updateGillespieProcess( GillespieProcessPtr aGillespieProcessPtr ) const
  {
    aGillespieProcessPtr->updateStepInterval( gsl_rng_uniform_pos( theRng ) );
  }


  GillespieProcessVectorCref getGillespieProcessVector() const
  {
    return theGillespieProcessVector;
  }


protected:

  GillespieProcessVector theGillespieProcessVector;

  NRPriorityQueue thePriorityQueue;

  //    GillespieProcessPtr    theLastProcess;

  Real            theTimeScale;

  Real            theTolerance;

  gsl_rng* const theRng;


};



#endif /* __NRSTEPPER_HPP */
