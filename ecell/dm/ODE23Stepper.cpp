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
    const UnsignedInt aSize( getReadOnlyVariableOffset() );

    theK1.resize( aSize );
  }

  bool ODE23Stepper::calculate()
  {
    const UnsignedInt aSize( getReadOnlyVariableOffset() );

    const Real eps_rel( getTolerance() );
    const Real eps_abs( getTolerance() * getAbsoluteToleranceFactor() );
    const Real a_y( getStateToleranceFactor() );
    const Real a_dydt( getDerivativeToleranceFactor() );

    const Real aCurrentTime( getCurrentTime() );

    // ========= 1 ===========
    fire();

    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );
	    
	// get k1
	const Real aVelocity( aVariable->getVelocity() );
	theK1[ c ] = aVelocity;
	
	// restore k1
	aVariable->loadValue( aVelocity * getStepInterval() 
			      + theValueBuffer[ c ] );

	// clear velocity
	aVariable->clearVelocity();
      }

    // ========= 2 ===========
    setCurrentTime( aCurrentTime + getStepInterval() );
    fire();

    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );
	    
	// get k2
	const Real aVelocity( aVariable->getVelocity() );
	theVelocityBuffer[ c ] = aVelocity;
	    
	aVariable->loadValue( ( aVelocity + theK1[ c ] ) * 0.25 
			      * getStepInterval() 
			      + theValueBuffer[ c ] );
	    
	// clear velocity
	aVariable->clearVelocity();
      }
	
    // ========= 3 ===========
    setCurrentTime( aCurrentTime + getStepInterval()*0.5 );
    fire();
	
    Real maxError( 0.0 );

    // restore theValueBuffer
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );
	
	const Real aVelocity( aVariable->getVelocity() );
	
	// ( k1 + k2 + k3 * 4 ) / 6 for ~Yn+1
	const Real anErrorEstimate( ( theK1[ c ] + theVelocityBuffer[ c ]
				      + aVelocity * 4.0 ) * ( 1.0 / 6.0 ) );

	// ( k1 + k2 ) / 2 for Yn+1
	theVelocityBuffer[ c ] += theK1[ c ];
	theVelocityBuffer[ c ] *= 0.5;

	const Real aTolerance( eps_rel *
				( a_y * fabs( theValueBuffer[ c ] ) 
				  +  a_dydt * fabs( theVelocityBuffer[ c ] ) )
				+ eps_abs );

	const Real anError( fabs( ( theVelocityBuffer[ c ] 
				    - anErrorEstimate ) / aTolerance ) );

	if( anError > maxError )
	  {
	    maxError = anError;
	  }

	// restore x (original value)
	aVariable->loadValue( theValueBuffer[ c ] );

	//// x(n+1) = x(n) + k2 * aStepInterval + O(h^3)
	aVariable->setVelocity( theVelocityBuffer[ c ] );
      }
    
    setMaxErrorRatio( maxError );

    // reset the stepper current time
    setCurrentTime( aCurrentTime );

    if ( maxError > 1.1 )
      {
	reset();
	setOriginalStepInterval( 0.0 );

	return false;
      }

    // set the error limit interval
    setOriginalStepInterval( getStepInterval() );

    return true;
  }

}
