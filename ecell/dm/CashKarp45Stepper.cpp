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

#include "CashKarp45Stepper.hpp"

namespace libecs
{

  LIBECS_DM_INIT( CashKarp45Stepper, Stepper );


  CashKarp45Stepper::CashKarp45Stepper()
  {
    ; // do nothing
  }
	    
  CashKarp45Stepper::~CashKarp45Stepper()
  {
    ; // do nothing
  }

  void CashKarp45Stepper::initialize()
  {
    AdaptiveDifferentialStepper::initialize();

    const UnsignedInt aSize( getReadOnlyVariableOffset() );

    theK1.resize( aSize );
    theK2.resize( aSize );
    theK3.resize( aSize );
    theK4.resize( aSize );
    theK5.resize( aSize );
    theK6.resize( aSize );

    theErrorEstimate.resize( aSize );
  }

  bool CashKarp45Stepper::calculate()
  {
    const UnsignedInt aSize( getReadOnlyVariableOffset() );

    const Real eps_rel( getTolerance() );
    const Real eps_abs( getTolerance() * getAbsoluteToleranceFactor() );
    const Real a_y( getStateToleranceFactor() );
    const Real a_dydt( getDerivativeToleranceFactor() );

    const Real aCurrentTime( getCurrentTime() );

    // ========= 1 ===========
    fireProcesses();

    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );
	    
	// get k1
	theK1[ c ] = aVariable->getVelocity();

	// restore k1 / 5 + x
	aVariable->loadValue( theK1[ c ] * .2  * getStepInterval()
			      + theValueBuffer[ c ] );

	// k1 * 37/378 for Yn+1
	theVelocityBuffer[ c ] = theK1[ c ] * ( 37.0 / 378.0 );
	// k1 * 2825/27648 for ~Yn+1
	theErrorEstimate[ c ] = theK1[ c ] * ( 2825.0 / 27648.0 );

	// clear velocity
	aVariable->clearVelocity();
      }

    // ========= 2 ===========
    setCurrentTime( aCurrentTime + getStepInterval()*0.2 );
    fireProcesses();

    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );
	
	theK2[ c ] = aVariable->getVelocity();
	    
	// restore k1 * 3/40+ k2 * 9/40 + x
	aVariable->
	  loadValue( ( theK1[ c ] * ( 3.0 / 40.0 ) 
		       + theK2[ c ] * ( 9.0 / 40.0 ) ) * getStepInterval()
		     + theValueBuffer[ c ] );
	    
	// k2 * 0 for Yn+1 (do nothing)
	//	    theVelocityBuffer[ c ] += theK2[ c ] * 0;
	// k2 * 0 for ~Yn+1 (do nothing)
	//	    theErrorEstimate[ c ] += theK2[ c ] * 0;
	    
	// clear velocity
	aVariable->clearVelocity();
      }
	
    // ========= 3 ===========
    setCurrentTime( aCurrentTime + getStepInterval()*0.3 );
    fireProcesses();
	
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );
	
	theK3[ c ] = aVariable->getVelocity();
	
	// restore k1 * 3/10 - k2 * 9/10 + k3 * 6/5 + x
	aVariable->
	  loadValue( ( theK1[ c ] * ( 3.0 / 10.0 )
		       - theK2[ c ] * ( 9.0 / 10.0 )
		       + theK3[ c ] * ( 6.0 / 5.0 ) ) * getStepInterval()
		     + theValueBuffer[ c ] );
	
	// k3 * 250/621 for Yn+1
	theVelocityBuffer[ c ] += theK3[ c ] * ( 250.0 / 621.0 );
	// k3 * 18575/48384 for ~Yn+1
	theErrorEstimate[ c ] += theK3[ c ] * ( 18575.0 / 48384.0 );
	
	// clear velocity
	aVariable->clearVelocity();
      }
    
    // ========= 4 ===========
    setCurrentTime( aCurrentTime + getStepInterval()*0.6 );
    fireProcesses();
    
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );
	
	theK4[ c ] = aVariable->getVelocity();
	
	// restore k2 * 5/2 - k1 * 11/54 - k3 * 70/27 + k4 * 35/27 + x
	aVariable->
	  loadValue( ( theK2[ c ] * ( 5.0 / 2.0 ) 
		       - theK1[ c ] * ( 11.0 / 54.0 )
		       - theK3[ c ] * ( 70.0 / 27.0 )
		       + theK4[ c ] * ( 35.0 / 27.0 ) ) * getStepInterval()
		     + theValueBuffer[ c ] );
	
	// k4 * 125/594 for Yn+1
	theVelocityBuffer[ c ] += theK4[ c ] * ( 125.0 / 594.0 );
	// k4 * 13525/55296 for ~Yn+1
	theErrorEstimate[ c ] += theK4[ c ] * ( 13525.0 / 55296.0 );
	    
	// clear velocity
	aVariable->clearVelocity();
      }
		
    // ========= 5 ===========
    setCurrentTime( aCurrentTime + getStepInterval() );
    fireProcesses();
	
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );
	
	theK5[ c ] = aVariable->getVelocity();
	    
	// restore k1 * 1631/55296 
	//         + k2 * 175/512 
	//         + k3 * 575/13824
	//         + k4 * 44275/110592
	//         + k5 * 253/4096
	aVariable->
	  loadValue( ( theK1[ c ] * ( 1631.0 / 55296.0 )
		       + theK2[ c ] * ( 175.0 / 512.0 )
		       + theK3[ c ] * ( 575.0 / 13824.0 )
		       + theK4[ c ] * ( 44275.0 / 110592.0 )
		       + theK5[ c ] * ( 253.0 / 4096.0 ) )
		     * getStepInterval() + theValueBuffer[ c ] );
	    
	// k5 * 0 for Yn+1(do nothing)
	//	    theVelocityBuffer[ c ] += theK5[ c ] * 0;
	// k5 * 277/14336 for ~Yn+1
	theErrorEstimate[ c ] += theK5[ c ] * ( 277.0 / 14336.0 );
	    
	// clear velocity
	aVariable->clearVelocity();
      }

    // ========= 6 ===========
    setCurrentTime( aCurrentTime + getStepInterval() * ( 7.0 / 8.0 ) );
    fireProcesses();
	
    Real maxError( 0.0 );
    
    // restore theValueBuffer
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );
	
	theK6[ c ] = aVariable->getVelocity();

	// k6 * 512/1771 for Yn+1
	theVelocityBuffer[ c ] += theK6[ c ] * ( 512.0 / 1771.0 );
	// k6 * 1/4 for ~Yn+1
	theErrorEstimate[ c ] += theK6[ c ] * .25;

	const Real 
	  aTolerance( eps_rel * 
		      ( a_y * fabs( theValueBuffer[ c ] ) 
			+ a_dydt * fabs( theVelocityBuffer[ c ] ) )
		      + eps_abs );
	    
	const Real 
	  anError( fabs( ( theVelocityBuffer[ c ] 
			   - theErrorEstimate[ c ] ) / aTolerance ) );
	    
	if( anError > maxError )
	  {
	    maxError = anError;
	  }

	// restore x (original value)
	aVariable->loadValue( theValueBuffer[ c ] );

	//// k1 * 37/378 + k3 * 250/621 + k4 * 125/594 + k6 * 512/1771)
	aVariable->setVelocity( theVelocityBuffer[ c ] );
      }

    setMaxErrorRatio( maxError );

    // reset the stepper current time
    setCurrentTime( aCurrentTime );

    if( maxError > 1.1 )
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
