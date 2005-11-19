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
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#include "Variable.hpp"

#include "ODE23Stepper.hpp"

LIBECS_DM_INIT( ODE23Stepper, Stepper );

ODE23Stepper::ODE23Stepper()
{
  ; // do nothing
}
	    
ODE23Stepper::~ODE23Stepper()
{
  ; // do nothing
}

void ODE23Stepper::initialize()
{
  AdaptiveDifferentialStepper::initialize();

  // theVelocityBuffer can be replaced by theK2
  // ODE23Stepper doesn't need it, but ODE45Stepper does for the efficiency 
}

void ODE23Stepper::interIntegrate2()
{
  Real const aCurrentTime( getCurrentTime() );

  for( VariableVector::size_type c( 0 );
       c != theVariableVector.size(); ++c )
    {
      VariablePtr const aVariable( theVariableVector[ c ] );

      aVariable->loadValue( theValueBuffer[ c ] );
      aVariable->interIntegrate( aCurrentTime );
    }
}

bool ODE23Stepper::calculate()
{
  const VariableVector::size_type aSize( getReadOnlyVariableOffset() );

  const Real eps_rel( getTolerance() );
  const Real eps_abs( getTolerance() * getAbsoluteToleranceFactor() );
  const Real a_y( getStateToleranceFactor() );
  const Real a_dydt( getDerivativeToleranceFactor() );

  const Real aCurrentTime( getCurrentTime() );
  const Real aStepInterval( getStepInterval() );

  theStateFlag = true;

  theTaylorSeries.reindex( 0 );

  // ========= 1 ===========
  interIntegrate2();
  fireProcesses();
  setVariableVelocity( theTaylorSeries[ 0 ] );

  // ========= 2 ===========
  setCurrentTime( aCurrentTime + aStepInterval );
  interIntegrate2();
  fireProcesses();
  setVariableVelocity( theTaylorSeries[ 1 ] );

  for( VariableVector::size_type c( 0 ); c < aSize; ++c )
    {
      theTaylorSeries[ 1 ][ c ] -= theTaylorSeries[ 0 ][ c ];
    }

  // ========= 3 ===========
  setCurrentTime( aCurrentTime + aStepInterval * 0.5 );
  interIntegrate2();
  fireProcesses();
  setVariableVelocity( theTaylorSeries[ 2 ] );

  Real maxError( 0.0 );

  // restore theValueBuffer
  for( VariableVector::size_type c( 0 ); c < aSize; ++c )
    {
      theTaylorSeries[ 1 ][ c ] *= 0.5;
      const Real anExpectedVelocity( theTaylorSeries[ 0 ][ c ]
      				     + theTaylorSeries[ 1 ][ c ] );

      // ( k1 + k2 + k3 * 4 ) / 6 for ~Yn+1
      // ( k1 + k2 - k3 * 2 ) / 3 for ( Yn+1 - ~Yn+1 ) as a local error
      const Real anEstimatedError
	( fabs( ( anExpectedVelocity - theTaylorSeries[ 2 ][ c ] ) 
		* ( 2.0 / 3.0 ) ) * aStepInterval );

      const Real aTolerance( eps_rel *
			     ( a_y * fabs( theValueBuffer[ c ] ) 
			       +  a_dydt * fabs( anExpectedVelocity ) * aStepInterval )
			     + eps_abs );

      //      const Real aTolerance( std::max( eps_rel * ( a_y * fabs( theValueBuffer[ c ] ) + 0.0 * fabs( anExpectedVelocity ) ), eps_abs ) );

      const Real anError( anEstimatedError / aTolerance );

      if( anError > maxError )
	{
	  maxError = anError;
	}

      // restore x (original value)
      theVariableVector[ c ]->loadValue( theValueBuffer[ c ] );
      theTaylorSeries[ 2 ][ c ] = 0.0;
    }
    
  setMaxErrorRatio( maxError );

  // reset the stepper current time
  setCurrentTime( aCurrentTime );
  resetAll();

  if ( maxError > 1.1 )
    {
      reset();
      return false;
    }

  // set the error limit interval
  return true;
}
