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

#include <time.h>

//#include <iostream>
#include <vector>
#include <algorithm>

#include <gsl/gsl_rng.h>

#include "libecs/libecs.hpp"
#include "libecs/DynamicPriorityQueue.hpp"
#include "libecs/Variable.hpp"
#include "libecs/Stepper.hpp"

USE_LIBECS;

  DECLARE_CLASS( NRProcess );

  DECLARE_VECTOR( Int,  IntVector );
//  DECLARE_VECTOR( Real, RealVector );
  DECLARE_VECTOR( NRProcessPtr, NRProcessVector );


  DECLARE_CLASS( NREvent );
  DECLARE_CLASS( NRStepper );


  // A pair of (reaction index, time) for inclusion in the priority queue.
  class NREvent
  {
  public:

    NREvent()
    {
      ; // do nothing
    }

    NREvent( RealCref aTime, NRProcessPtr aProcess )
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

    NRProcessPtr getProcess() const
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
    NRProcessPtr theProcess;


   };



  DECLARE_TYPE( DynamicPriorityQueue<NREvent>, NRPriorityQueue );


  class NRStepper 
    : 
    public DiscreteEventStepper 
  {

    LIBECS_DM_OBJECT( Stepper, NRStepper );

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

    NRProcessVectorCref getNRProcessVector() const
    {
      return theNRProcessVector;
    }


  protected:

    NRProcessVector theNRProcessVector;

    NRPriorityQueue thePriorityQueue;

    //    NRProcessPtr    theLastProcess;

    Real            theTimeScale;

    Real            theTolerance;

    gsl_rng* const theRng;


  };



#endif /* __NRSTEPPER_HPP */
