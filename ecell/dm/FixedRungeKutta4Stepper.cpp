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

#include "FixedRungeKutta4Stepper.hpp"

LIBECS_DM_INIT( FixedRungeKutta4Stepper, Stepper );

namespace libecs
{

  FixedRungeKutta4Stepper::FixedRungeKutta4Stepper()
  {
    ; // do nothing
  }
	    
  FixedRungeKutta4Stepper::~FixedRungeKutta4Stepper()
  {
    ; // do nothing
  }

  void FixedRungeKutta4Stepper::step()
  {
    // clear
    clearVariables();

    const Real aCurrentTime( getCurrentTime() );

    // ========= 1 ===========
    fireProcesses();

    const UnsignedInt aSize( getReadOnlyVariableOffset() );
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );

	// get k1
	Real aVelocity( aVariable->getVelocity() );

	// restore k1 / 2 + x
	aVariable->loadValue( aVelocity * .5 * getStepInterval()
			      + theValueBuffer[ c ] );

	theVelocityBuffer[ c ] = aVelocity;

	// clear velocity
	aVariable->clearVelocity();
      }

    // ========= 2 ===========
    setCurrentTime( aCurrentTime + getStepInterval()*0.5 );
    fireProcesses();

    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );

	const Real aVelocity( aVariable->getVelocity() );
	theVelocityBuffer[ c ] += aVelocity + aVelocity;

	// restore k2 / 2 + x
	aVariable->loadValue( aVelocity * .5 * getStepInterval()
			      + theValueBuffer[ c ] );


	// clear velocity
	aVariable->clearVelocity();
      }

    // ========= 3 ===========
    //    setCurrentTime( aCurrentTime + getStepInterval()*0.5 );
    fireProcesses();
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );

	const Real aVelocity( aVariable->getVelocity() );
	theVelocityBuffer[ c ] += aVelocity + aVelocity;

	// restore k3 + x
	aVariable->loadValue( aVelocity * getStepInterval()
			      + theValueBuffer[ c ] );

	// clear velocity
	aVariable->clearVelocity();
      }

    // ========= 4 ===========
    setCurrentTime( aCurrentTime + getStepInterval() );
    fireProcesses();

    // restore theValueBuffer
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );

	const Real aVelocity( aVariable->getVelocity() );

	// restore x (original value)
	aVariable->loadValue( theValueBuffer[ c ] );

	//// x(n+1) = x(n) + 1/6 * (k1 + k4 + 2 * (k2 + k3)) + O(h^5)

	theVelocityBuffer[ c ] += aVelocity;
	theVelocityBuffer[ c ] *= ( 1.0 / 6.0 );
	aVariable->setVelocity( theVelocityBuffer[ c ] );
      }

    // reset the stepper current time
    setCurrentTime( aCurrentTime );

    // set the error limit interval
    setOriginalStepInterval( getStepInterval() );
  }

}
