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
// written by Kouichi Takahashi <shafi@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//

#include "Variable.hpp"

#include "ODE23Stepper.hpp"

LIBECS_DM_INIT( ODE23Stepper, Stepper );

namespace libecs
{

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

    // the number of write variables
    const VariableVector::size_type aSize( getReadOnlyVariableOffset() );

    theK1.resize( aSize );
    theK2.resize( aSize );

    // theVelocityBuffer can be replaced by theK2
    // ODE23Stepper doesn't need it, but ODE45Stepper does for the efficiency 
    theVelocityBuffer.clear();
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

    // ========= 1 ===========
    interIntegrate2();
    fireProcesses();

    for( VariableVector::size_type c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );
	    
	// get k1
	const Real aVelocity( aVariable->getVelocity() );
	theK1[ c ] = aVelocity;
	
	// clear velocity
	aVariable->clearVelocity();
      }

    // ========= 2 ===========
    setCurrentTime( aCurrentTime + getStepInterval() );
    interIntegrate2();
    fireProcesses();

    for( VariableVector::size_type c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );
	    
	// get k2
	const Real aVelocity( aVariable->getVelocity() );
	theK2[ c ] = aVelocity - theK1[ c ];
    
	// clear velocity
	aVariable->clearVelocity();
      }
	
    // ========= 3 ===========
    setCurrentTime( aCurrentTime + getStepInterval() * 0.5 );
    interIntegrate2();
    fireProcesses();
	
    Real maxError( 0.0 );

    // restore theValueBuffer
    for( VariableVector::size_type c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );
	
	const Real aVelocity( aVariable->getVelocity() );
 
	// ( k1 - k2 ) / 2 for K2
	theK2[ c ] *= 0.5;

	const Real anExpectedVelocity( theK1[ c ] + theK2[ c ] );

	// ( k1 + k2 + k3 * 4 ) / 6 for ~Yn+1
	// ( k1 + k2 - k3 * 2 ) / 3 for ( Yn+1 - ~Yn+1 ) as a local error
	const Real anEstimatedError
	  ( fabs( ( anExpectedVelocity - aVelocity ) * ( 2.0 / 3.0 ) ) );

	const Real aTolerance( eps_rel *
				( a_y * fabs( theValueBuffer[ c ] ) 
				  +  a_dydt * fabs( anExpectedVelocity ) )
				+ eps_abs );

	const Real anError( anEstimatedError / aTolerance );

	if( anError > maxError )
	  {
	    maxError = anError;
	  }

	// restore x (original value)
	aVariable->loadValue( theValueBuffer[ c ] );

	//// x(n+1) = x(n) + k2 * aStepInterval + O(h^3)
	aVariable->setVelocity( anExpectedVelocity );
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

}
