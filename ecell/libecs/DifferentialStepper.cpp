//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Kouichi Takahashi <shafi@e-cell.org>,
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//

#include <limits>

#include "Util.hpp"
#include "Variable.hpp"
#include "Interpolant.hpp"
#include "Process.hpp"
#include "Model.hpp"

#include "DifferentialStepper.hpp"


namespace libecs
{

  LIBECS_DM_INIT_STATIC( DifferentialStepper, Stepper );
  LIBECS_DM_INIT_STATIC( AdaptiveDifferentialStepper, Stepper );


  DifferentialStepper::DifferentialStepper()
    :
    theNextStepInterval( 0.001 ),
    theStateFlag( false )
  {
    ; // do nothing
  }

  DifferentialStepper::~DifferentialStepper()
  {
    ; // do nothing
  }

  void DifferentialStepper::initialize()
  {
    Stepper::initialize();

    createVariableProxies();

    // size of the velocity buffer == the number of *write* variables.
    theVelocityBuffer.resize( getReadOnlyVariableOffset() );

    // should create another method for property slot ?
    //    setNextStepInterval( getStepInterval() );

    //    theStateFlag = false;

    
  }

  void DifferentialStepper::reset()
  {
    // clear velocity buffer
    theVelocityBuffer.assign( theVelocityBuffer.size(), 0.0 );

    Stepper::reset();
  }

  void DifferentialStepper::resetAll()
  {
    const VariableVector::size_type aSize( theVariableVector.size() );
    for( VariableVector::size_type c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );
	aVariable->loadValue( theValueBuffer[ c ] );
      }
  }

  void DifferentialStepper::interIntegrate()
  {
    Real const aCurrentTime( getCurrentTime() );

    VariableVector::size_type c( theReadWriteVariableOffset );
    for( ; c != theReadOnlyVariableOffset; ++c )
      {
      	VariablePtr const aVariable( theVariableVector[ c ] );

	aVariable->interIntegrate( aCurrentTime );
      }

    // RealOnly Variables must be reset by the values in theValueBuffer
    // before interIntegrate().
    for( ; c != theVariableVector.size(); ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );

	aVariable->loadValue( theValueBuffer[ c ] );
	aVariable->interIntegrate( aCurrentTime );
      }

  }

  void DifferentialStepper::interrupt( StepperPtr const aCaller )
  {
    const Real aCallerTimeScale( aCaller->getTimeScale() );
    const Real aStepInterval( getStepInterval() );

    // If the step size of this is less than caller's timescale,
    // ignore this interruption.
    if( aCallerTimeScale >= aStepInterval )
      {
	return;
      }

    // if all Variables didn't change its value more than 10%,
    // ignore this interruption.

    /*  !!! currently this is disabled

    if( checkExternalError() )
      {
	return;
      }
    */
	
    const Real aCurrentTime( getCurrentTime() );
    const Real aCallerCurrentTime( aCaller->getCurrentTime() );

    // aCallerTimeScale == 0 implies need for immediate reset
    if( aCallerTimeScale != 0.0 )
      {
	// Shrink the next step size to that of caller's
	setNextStepInterval( aCallerTimeScale );

	const Real aNextStep( aCurrentTime + aStepInterval );
	const Real aCallerNextStep( aCallerCurrentTime + aCallerTimeScale );

	// If the next step of this occurs *before* the next step 
	// of the caller, just shrink step size of this Stepper.
	if( aNextStep <= aCallerNextStep )
	  {
	    //	    std::cerr << aCurrentTime << " return" << std::endl;

	    return;
	  }

	//	std::cerr << aCurrentTime << " noreset" << std::endl;

	
	// If the next step of this will occur *after* the caller,
	// reschedule this Stepper, as well as shrinking the next step size.
	
	//    setStepInterval( aCallerCurrentTime + ( aCallerTimeScale * 0.5 ) 
	//		     - aCurrentTime );
      }
    else
      {
	// reset step interval to the default
	
	//	std::cerr << aCurrentTime << " reset" << std::endl;

	setNextStepInterval( 0.001 );
      }
      
    loadStepInterval( aCallerCurrentTime - aCurrentTime );
    getModel()->reschedule( this );
  }


  ////////////////////////// AdaptiveDifferentialStepper

  AdaptiveDifferentialStepper::AdaptiveDifferentialStepper()
    :
    theTolerance( 1.0e-6 ),
    theAbsoluteToleranceFactor( 1.0 ),
    theStateToleranceFactor( 1.0 ),
    theDerivativeToleranceFactor( 1.0 ),
    theEpsilonChecked( 0 ),
    theAbsoluteEpsilon( 0.1 ),
    theRelativeEpsilon( 0.1 ),
    safety( 0.9 ),
    theMaxErrorRatio( 1.0 )
  {
    // use more narrow range
    setMinStepInterval( 1e-100 );
    setMaxStepInterval( 1e+100 );
  }

  AdaptiveDifferentialStepper::~AdaptiveDifferentialStepper()
  {
    ; // do nothing
  }


  void AdaptiveDifferentialStepper::initialize()
  {
    DifferentialStepper::initialize();

    theEpsilonChecked = ( theEpsilonChecked 
			  || ( theDependentStepperVector.size() > 1 ) );
  }

  void AdaptiveDifferentialStepper::step()
  {
    theStateFlag = false;

    clearVariables();

    setStepInterval( getNextStepInterval() );
    //    setTolerableInterval( 0.0 );

    while ( !calculate() )
      {
	const Real anExpectedStepInterval( safety * getStepInterval() 
					   * pow( getMaxErrorRatio(),
						  -1.0 / getOrder() ) );
	//	const Real anExpectedStepInterval( 0.5 * getStepInterval() );

	if ( anExpectedStepInterval > getMinStepInterval() )
	  {
	    // shrink it if the error exceeds 110%
	    setStepInterval( anExpectedStepInterval );

	    //	    std::cerr << "s " << getCurrentTime() 
	    //		      << ' ' << getStepInterval()
	    //		      << std::endl;
	  }
	else
	  {
	    setStepInterval( getMinStepInterval() );

	    // this must return false,
	    // so theTolerableStepInterval does NOT LIMIT the error.
	    THROW_EXCEPTION( SimulationError,
			     "The error-limit step interval of Stepper [" + 
			     getID() + "] is too small." );
 
	    calculate();
	    break;
	  }
      }

    setTolerableStepInterval( getStepInterval() );

    theStateFlag = true;

    // grow it if error is 50% less than desired
    const Real maxError( getMaxErrorRatio() );
    if ( maxError < 0.5 )
      {
	const Real aNewStepInterval( getTolerableStepInterval() * safety
				     * pow( maxError ,
					    -1.0 / ( getOrder() + 1 ) ) );
	//	const Real aNewStepInterval( getStepInterval() * 2.0 );

	setNextStepInterval( aNewStepInterval );

	//	std::cerr << "g " << getCurrentTime() << ' ' 
	//		  << getStepInterval() << std::endl;
      }
    else 
      {
	setNextStepInterval( getTolerableStepInterval() );
      }

    // check the tolerances for Epsilon
    if ( isEpsilonChecked() ) {
      const VariableVector::size_type aSize( getReadOnlyVariableOffset() );

      for ( VariableVector::size_type c( 0 ); c < aSize; ++c )
	{
	  VariablePtr const aVariable( theVariableVector[ c ] );

	  const Real aTolerance( FMA( fabs( aVariable->getValue() ),
				      theRelativeEpsilon,
				      theAbsoluteEpsilon ) );

	  const Real aVelocity( fabs( theVelocityBuffer[ c ] ) );

	  if ( aTolerance < aVelocity * getStepInterval() )
	    {
	      setStepInterval( aTolerance / aVelocity );
	    }
	}
    }

    //    std::cout << getCurrentTime() << "\t"
    //	      << getStepInterval() << std::endl;
  }

} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
