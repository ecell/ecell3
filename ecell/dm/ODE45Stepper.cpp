//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2002 Keio University
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

#include "Variable.hpp"

#include "ODE45Stepper.hpp"

LIBECS_DM_INIT( ODE45Stepper, Stepper );


ODE45Stepper::ODE45Stepper()
  :
  theInterrupted( true ),
  theStiffness( 1.0 )
{
  ; // do nothing
}
	    
ODE45Stepper::~ODE45Stepper()
{
  ; // do nothing
}

void ODE45Stepper::initialize()
{
  AdaptiveDifferentialStepper::initialize();

  const VariableVector::size_type aSize( getReadOnlyVariableOffset() );

  theK1.resize( aSize );
  theK2.resize( aSize );
  theK3.resize( aSize );
  theK4.resize( aSize );
  theK5.resize( aSize );
  theK6.resize( aSize );
  theK7.resize( aSize );

  theMidVelocityBuffer.resize( aSize );

  theInterrupted = true;
}

void ODE45Stepper::step()
{
  AdaptiveDifferentialStepper::step();

  //   check if the step interval was changed, by epsilon
  if ( fabs( getTolerableStepInterval() - getStepInterval() )
       > std::numeric_limits<Real>::epsilon() )
    {
      theInterrupted = true;
    }
}

bool ODE45Stepper::calculate()
{
  const VariableVector::size_type aSize( getReadOnlyVariableOffset() );

  const Real eps_rel( getTolerance() );
  const Real eps_abs( getTolerance() * getAbsoluteToleranceFactor() );
  const Real a_y( getStateToleranceFactor() );
  const Real a_dydt( getDerivativeToleranceFactor() );

  const Real aCurrentTime( getCurrentTime() );

  // ========= 1 ===========

  if ( theInterrupted )
    //    if ( 1 )
    {
      interIntegrate();
      fireProcesses();

      for( VariableVector::size_type c( 0 ); c < aSize; ++c )
	{
	  VariablePtr const aVariable( theVariableVector[ c ] );
	
	  // get k1
	  theK1[ c ] = aVariable->getVelocity();
	    
	  aVariable->loadValue( theK1[ c ] * ( 1.0 / 5.0 )
				* getStepInterval()
				+ theValueBuffer[ c ] );

	  // clear velocity
	  aVariable->clearVelocity();
	}
    }
  else
    {
      for( VariableVector::size_type c( 0 ); c < aSize; ++c )
	{
	  VariablePtr const aVariable( theVariableVector[ c ] );
	
	  // get k1
	  theK1[ c ] = theK7[ c ];

	  aVariable->loadValue( theK1[ c ] * ( 1.0 / 5.0 )
				* getStepInterval()
				+ theValueBuffer[ c ] );

	  // clear velocity
	  aVariable->clearVelocity();
	}	
    }

  // ========= 2 ===========
  setCurrentTime( aCurrentTime + getStepInterval() * 0.2 );
  interIntegrate();
  fireProcesses();

  for( VariableVector::size_type c( 0 ); c < aSize; ++c )
    {
      VariablePtr const aVariable( theVariableVector[ c ] );
	
      // get k2
      theK2[ c ] = aVariable->getVelocity();

      aVariable->loadValue( ( theK1[ c ] * ( 3.0 / 40.0 ) 
			      + theK2[ c ] * ( 9.0 / 40.0 ) )
			    * getStepInterval()
			    + theValueBuffer[ c ] );

      // clear velocity
      aVariable->clearVelocity();
    }


  // ========= 3 ===========
  setCurrentTime( aCurrentTime + getStepInterval() * 0.3 );
  interIntegrate();
  fireProcesses();

  for( VariableVector::size_type c( 0 ); c < aSize; ++c )
    {
      VariablePtr const aVariable( theVariableVector[ c ] );
	
      // get k3
      theK3[ c ] = aVariable->getVelocity();

      aVariable->loadValue( ( theK1[ c ] * ( 44.0 / 45.0 ) 
			      - theK2[ c ] * ( 56.0 / 15.0 )
			      + theK3[ c ] * ( 32.0 / 9.0 ) )
			    * getStepInterval()
			    + theValueBuffer[ c ] );

      // clear velocity
      aVariable->clearVelocity();
    }

  // ========= 4 ===========
  setCurrentTime( aCurrentTime + getStepInterval() * 0.8 );
  interIntegrate();
  fireProcesses();

  for( VariableVector::size_type c( 0 ); c < aSize; ++c )
    {
      VariablePtr const aVariable( theVariableVector[ c ] );
	
      // get k4
      theK4[ c ] = aVariable->getVelocity();

      aVariable->loadValue( ( theK1[ c ] * ( 19372.0 / 6561.0 ) 
			      - theK2[ c ] * ( 25360.0 / 2187.0 )
			      + theK3[ c ] * ( 64448.0 / 6561.0 )
			      - theK4[ c ] * ( 212.0 / 729.0 ) )
			    * getStepInterval()
			    + theValueBuffer[ c ] );

      // clear velocity
      aVariable->clearVelocity();
    }

  // ========= 5 ===========
  setCurrentTime( aCurrentTime + getStepInterval() * ( 8.0 / 9.0 ) );
  interIntegrate();
  fireProcesses();

  for( VariableVector::size_type c( 0 ); c < aSize; ++c )
    {
      VariablePtr const aVariable( theVariableVector[ c ] );
	
      // get k5
      theK5[ c ] = aVariable->getVelocity();

      // temporarily set Y^6
      theMidVelocityBuffer[ c ] 
	= theK1[ c ] * ( 9017.0 / 3168.0 ) 
	- theK2[ c ] * ( 355.0 / 33.0 )
	+ theK3[ c ] * ( 46732.0 / 5247.0 )
	+ theK4[ c ] * ( 49.0 / 176.0 )
	- theK5[ c ] * ( 5103.0 / 18656.0 );
 
      aVariable->loadValue( theMidVelocityBuffer[ c ] * getStepInterval()
			    + theValueBuffer[ c ] );

      // clear velocity
      aVariable->clearVelocity();
    }

  // ========= 6 ===========

  // estimate stiffness
  Real aDenominator( 0.0 );
  Real aLipschitzConstant( 0.0 );

  setCurrentTime( aCurrentTime + getStepInterval() );
  interIntegrate();
  fireProcesses();

  for( VariableVector::size_type c( 0 ); c < aSize; ++c )
    {
      VariablePtr const aVariable( theVariableVector[ c ] );
	
      // get k6
      theK6[ c ] = aVariable->getVelocity();

      theVelocityBuffer[ c ] 
	= theK1[ c ] * ( 35.0 / 384.0 )
	// + theK2[ c ] * 0.0
	+ theK3[ c ] * ( 500.0 / 1113.0 )
	+ theK4[ c ] * ( 125.0 / 192.0 )
	+ theK5[ c ] * ( -2187.0 / 6784.0 )
	+ theK6[ c ] * ( 11.0 / 84.0 );

      aDenominator
	+= ( theVelocityBuffer[ c ] - theMidVelocityBuffer[ c ] )
	* ( theVelocityBuffer[ c ] - theMidVelocityBuffer[ c ] );

      aVariable->loadValue( theVelocityBuffer[ c ] * getStepInterval()
			    + theValueBuffer[ c ] );

      // clear velocity
      aVariable->clearVelocity();
    }

  // ========= 7 ===========
  setCurrentTime( aCurrentTime + getStepInterval() );
  interIntegrate();
  fireProcesses();

  // evaluate error
  Real maxError( 0.0 );

  for( VariableVector::size_type c( 0 ); c < aSize; ++c )
    {
      VariablePtr const aVariable( theVariableVector[ c ] );

      // get k7
      theK7[ c ] = aVariable->getVelocity();

      // calculate error
      const Real anEstimatedError( theK1[ c ] * ( 71.0 / 57600.0 )
				   + theK3[ c ] * ( -71.0 / 16695.0 )
				   + theK4[ c ] * ( 71.0 / 1920.0 )
				   + theK5[ c ] * ( -17253.0 / 339200.0 )
				   + theK6[ c ] * ( 22.0 / 525.0 )
				   + theK7[ c ] * ( -1.0 / 40.0 ) );

      // 	const Real anEstimatedError( theK1[ c ] * ( 71.0 / 86400.0 )
      // 				     + theK3[ c ] * ( -142.0 / 50085.0 )
      // 				     + theK4[ c ] * ( 71.0 / 2880.0 )
      // 				     + theK5[ c ] * ( -5751.0 / 169600.0 )
      // 				     + theK6[ c ] * ( 44.0 / 1575.0 )
      // 				     + theK7[ c ] * ( -1.0 / 60.0 ) );

      aLipschitzConstant 
	+= ( theK7[ c ] - theK6[ c ] ) * ( theK7[ c ] - theK6[ c ] );

      // calculate velocity for Xn+.5
      theMidVelocityBuffer[ c ] 
	= theK1[ c ] * ( 6025192743.0 / 30085553152.0 )
	+ theK3[ c ] * ( 51252292925.0 / 65400821598.0 )
	+ theK4[ c ] * ( -2691868925.0 / 45128329728.0 )
	+ theK5[ c ] * ( 187940372067.0 / 1594534317056.0 )
	+ theK6[ c ] * ( -1776094331.0 / 19743644256.0 )
	+ theK7[ c ] * ( 11237099.0 / 235043384.0 );

      const Real aTolerance( eps_rel * 
			     ( a_y * fabs( theValueBuffer[ c ] ) 
			       + a_dydt * fabs( theVelocityBuffer[ c ] ) )
			     + eps_abs );

      const Real anError( fabs( anEstimatedError / aTolerance ) );

      if( anError > maxError )
	{
	  maxError = anError;
	}
	
      aVariable->setVelocity( theVelocityBuffer[ c ] );
    }

  aLipschitzConstant /= aDenominator;
  aLipschitzConstant  = sqrt( aLipschitzConstant );
  setStiffness( aLipschitzConstant / 3.3 );

  //    std::cout << getCurrentTime() << "\t"
  //	      << getStiffness() << std::endl;

  resetAll(); // reset all value

  setMaxErrorRatio( maxError );
  setCurrentTime( aCurrentTime );

  if( maxError > 1.1 )
    {
      // reset the stepper current time
      reset();
      theInterrupted = true;
      return false;
    }

  // set the error limit interval
  theInterrupted = false;

  return true;
}

void ODE45Stepper::interrupt( StepperPtr aCaller )
{
  theInterrupted = true;
  AdaptiveDifferentialStepper::interrupt( aCaller );
}


