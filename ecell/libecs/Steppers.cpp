//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
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

#include "Steppers.hpp"

namespace libecs
{


  ////////////////////////// FixedEuler1Stepper


  FixedEuler1Stepper::FixedEuler1Stepper()
  {
    ; // do nothing
  }

  void FixedEuler1Stepper::step()
  {
    const UnsignedInt aSize( getReadOnlyVariableOffset() );

    clear();

    process();

    setStepInterval( getNextStepInterval() );

    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );

	theVelocityBuffer[ c ] = aVariable->getVelocity();

	// avoid negative value
	while( aVariable->checkRange( getStepInterval() ) == false )
	  {
	    // don't use setStepInterval()
	    //FIXME:
	    loadStepInterval( getStepInterval() * 0.5 );
	  }
      }

    if( getStepInterval() < getTolerantStepInterval() )
      {
  	setNextStepInterval( getStepInterval() * 2.0 );
      }
    else 
      {
	setNextStepInterval( getStepInterval() );
      }
  }


  ////////////////////////// FixedRungeKutta4Stepper


  FixedRungeKutta4Stepper::FixedRungeKutta4Stepper()
  {
    ; // do nothing
  }

  void FixedRungeKutta4Stepper::step()
  {
    // clear
    clear();

    const Real aCurrentTime( getCurrentTime() );

    // ========= 1 ===========
    process();

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
	aVariable->setVelocity( 0.0 );
      }

    // ========= 2 ===========
    setCurrentTime( aCurrentTime + getStepInterval()*0.5 );
    process();

    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );

	const Real aVelocity( aVariable->getVelocity() );
	theVelocityBuffer[ c ] += aVelocity + aVelocity;

	// restore k2 / 2 + x
	aVariable->loadValue( aVelocity * .5 * getStepInterval()
			      + theValueBuffer[ c ] );


	// clear velocity
	aVariable->setVelocity( 0.0 );
      }

    // ========= 3 ===========
    //    setCurrentTime( aCurrentTime + getStepInterval()*0.5 );
    process();
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );

	const Real aVelocity( aVariable->getVelocity() );
	theVelocityBuffer[ c ] += aVelocity + aVelocity;

	// restore k3 + x
	aVariable->loadValue( aVelocity * getStepInterval()
			      + theValueBuffer[ c ] );

	// clear velocity
	aVariable->setVelocity( 0.0 );
      }

    // ========= 4 ===========
    setCurrentTime( aCurrentTime + getStepInterval() );
    process();

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


  ////////////////////////// Fehlberg23Stepper


  Fehlberg23Stepper::Fehlberg23Stepper()
  {
    ; // do nothing
  }

  void Fehlberg23Stepper::initialize()
  {
    AdaptiveDifferentialStepper::initialize();

    // the number of write variables
    const UnsignedInt aSize( getReadOnlyVariableOffset() );

    theK1.resize( aSize );
  }

  bool Fehlberg23Stepper::calculate()
  {
    const UnsignedInt aSize( getReadOnlyVariableOffset() );

    const Real eps_rel( getTolerance() );
    const Real eps_abs( getTolerance() * getAbsoluteToleranceFactor() );
    const Real a_y( getStateToleranceFactor() );
    const Real a_dydt( getDerivativeToleranceFactor() );

    const Real aCurrentTime( getCurrentTime() );

    // ========= 1 ===========
    process();

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
	aVariable->setVelocity( 0.0 );
      }

    // ========= 2 ===========
    setCurrentTime( aCurrentTime + getStepInterval() );
    process();

    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );
	    
	// get k2
	const Real aVelocity( aVariable->getVelocity() );
	theVelocityBuffer[ c ] = aVelocity;
	    
	// restore (k1 + k2) / 2
	//	    aVariable->loadValue( ( aVelocity + aVelocity 
	//				    - theK1[ c ] ) * getStepInterval()
	//				  + theValueBuffer[ c ] );
	    
	aVariable->loadValue( ( aVelocity + theK1[ c ] ) * 0.25 
			      * getStepInterval() 
			      + theValueBuffer[ c ] );
	    
	// clear velocity
	aVariable->setVelocity( 0.0 );
      }
	
    // ========= 3 ===========
    setCurrentTime( aCurrentTime + getStepInterval()*0.5 );
    process();
	
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


  ////////////////////////// CashKarp45Stepper


  CashKarp45Stepper::CashKarp45Stepper()
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
    process();

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
	aVariable->setVelocity( 0.0 );
      }

    // ========= 2 ===========
    setCurrentTime( aCurrentTime + getStepInterval()*0.2 );
    process();

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
	aVariable->setVelocity( 0.0 );
      }
	
    // ========= 3 ===========
    setCurrentTime( aCurrentTime + getStepInterval()*0.3 );
    process();
	
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
	aVariable->setVelocity( 0.0 );
      }
    
    // ========= 4 ===========
    setCurrentTime( aCurrentTime + getStepInterval()*0.6 );
    process();
    
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
	aVariable->setVelocity( 0.0 );
      }
		
    // ========= 5 ===========
    setCurrentTime( aCurrentTime + getStepInterval() );
    process();
	
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
	aVariable->setVelocity( 0.0 );
      }

    // ========= 6 ===========
    setCurrentTime( aCurrentTime + getStepInterval() * ( 7.0 / 8.0 ) );
    process();
	
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


  ////////////////////////// DormandPrince547MStepper


  DormandPrince547MStepper::DormandPrince547MStepper()
    :
    theInterrupted( true )
  {
    ; // do nothing
  }

  void DormandPrince547MStepper::initialize()
  {
    AdaptiveDifferentialStepper::initialize();

    const UnsignedInt aSize( getReadOnlyVariableOffset() );

    theK1.resize( aSize );
    theK2.resize( aSize );
    theK3.resize( aSize );
    theK4.resize( aSize );
    theK5.resize( aSize );
    theK6.resize( aSize );
    theK7.resize( aSize );

    theMidVelocityBuffer.resize( aSize );
    theErrorEstimate.resize( aSize );

    theInterrupted = true;
  }

  void DormandPrince547MStepper::step()
  {
    AdaptiveDifferentialStepper::step();

    if ( getOriginalStepInterval() > getStepInterval() )
      {
	theInterrupted = true;
      }
  }

  bool DormandPrince547MStepper::calculate()
  {
    const UnsignedInt aSize( getReadOnlyVariableOffset() );

    const Real eps_rel( getTolerance() );
    const Real eps_abs( getTolerance() * getAbsoluteToleranceFactor() );
    const Real a_y( getStateToleranceFactor() );
    const Real a_dydt( getDerivativeToleranceFactor() );

    const Real aCurrentTime( getCurrentTime() );

    // ========= 1 ===========

    if ( theInterrupted )
    //    if ( 1 )
      {
	process();

	for( UnsignedInt c( 0 ); c < aSize; ++c )
	  {
	    VariablePtr const aVariable( theVariableVector[ c ] );
	
	    // get k1
	    theK1[ c ] = aVariable->getVelocity();
	    
	    aVariable->loadValue( theK1[ c ] * ( 1.0 / 5.0 )
				  * getStepInterval()
				  + theValueBuffer[ c ] );

	    // k1 * 35/384 for Yn+1
	    theVelocityBuffer[ c ] = theK1[ c ] * ( 35.0 / 384.0 );
	    // k1 * 5179/57600 for ~Yn+1
	    theErrorEstimate[ c ] = theK1[ c ] * ( 5179.0 / 57600.0 );

	    // clear velocity
	    aVariable->setVelocity( 0.0 );
	  }
      }
    else
      {
	for( UnsignedInt c( 0 ); c < aSize; ++c )
	  {
	    VariablePtr const aVariable( theVariableVector[ c ] );
	
	    // get k1
	    theK1[ c ] = theK7[ c ];

	    aVariable->loadValue( theK1[ c ] * ( 1.0 / 5.0 )
				  * getStepInterval()
				  + theValueBuffer[ c ] );

	    // k1 * 35/384 for Yn+1
	    theVelocityBuffer[ c ] = theK1[ c ] * ( 35.0 / 384.0 );
	    // k1 * 5179/57600 for ~Yn+1
	    theErrorEstimate[ c ] = theK1[ c ] * ( 5179.0 / 57600.0 );

	    // clear velocity
	    aVariable->setVelocity( 0.0 );
	  }	
      }

    // ========= 2 ===========
    setCurrentTime( aCurrentTime + getStepInterval() * 0.2 );
    process();

    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );
	
	// get k2
	theK2[ c ] = aVariable->getVelocity();

	aVariable->loadValue( ( theK1[ c ] * ( 3.0 / 40.0 ) 
				+ theK2[ c ] * ( 9.0 / 40.0 ) )
			      * getStepInterval()
			      + theValueBuffer[ c ] );

	// k2 * 0 for Yn+1 (do nothing)
	//	    theVelocityBuffer[ c ] += theK2[ c ] * 0;
	// k2 * 0 for ~Yn+1 (do nothing)
	//	    theErrorEstimate[ c ] += theK2[ c ] * 0;

	// clear velocity
	aVariable->setVelocity( 0.0 );
      }


    // ========= 3 ===========
    setCurrentTime( aCurrentTime + getStepInterval() * 0.3 );
    process();

    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );
	
	// get k3
	theK3[ c ] = aVariable->getVelocity();

	aVariable->loadValue( ( theK1[ c ] * ( 44.0 / 45.0 ) 
				- theK2[ c ] * ( 56.0 / 15.0 )
				+ theK3[ c ] * ( 32.0 / 9.0 ) )
			      * getStepInterval()
			      + theValueBuffer[ c ] );

	// k3 * 500/1113 for Yn+1
	theVelocityBuffer[ c ] += theK3[ c ] * ( 500.0 / 1113.0 );
	// k3 * 7571/16695 for ~Yn+1
	theErrorEstimate[ c ] += theK3[ c ] * ( 7571.0 / 16695.0 );

	// clear velocity
	aVariable->setVelocity( 0.0 );
      }

    // ========= 4 ===========
    setCurrentTime( aCurrentTime + getStepInterval() * 0.8 );
    process();

    for( UnsignedInt c( 0 ); c < aSize; ++c )
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

	// k4 * 125/192 for Yn+1
	theVelocityBuffer[ c ] += theK4[ c ] * ( 125.0 / 192.0 );
	// k4 * 393/640 for ~Yn+1
	theErrorEstimate[ c ] += theK4[ c ] * ( 393.0 / 640.0 );

	// clear velocity
	aVariable->setVelocity( 0.0 );
      }

    // ========= 5 ===========
    setCurrentTime( aCurrentTime + getStepInterval() * ( 8.0 / 9.0 ) );
    process();

    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );
	
	// get k5
	theK5[ c ] = aVariable->getVelocity();

	aVariable->loadValue( ( theK1[ c ] * ( 9017.0 / 3168.0 ) 
				- theK2[ c ] * ( 355.0 / 33.0 )
				+ theK3[ c ] * ( 46732.0 / 5247.0 )
				+ theK4[ c ] * ( 49.0 / 176.0 )
				- theK5[ c ] * ( 5103.0 / 18656.0 ) )
			      * getStepInterval()
			      + theValueBuffer[ c ] );

	// k5 * -2187/6784 for Yn+1
	theVelocityBuffer[ c ] += theK5[ c ] * ( -2187.0 / 6784.0 );
	// k5 * -92097/339200 for ~Yn+1
	theErrorEstimate[ c ] += theK5[ c ] * ( -92097.0 / 339200.0 );

	// clear velocity
	aVariable->setVelocity( 0.0 );
      }

    // ========= 6 ===========
    setCurrentTime( aCurrentTime + getStepInterval() );
    process();

    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );
	
	// get k6
	theK6[ c ] = aVariable->getVelocity();

	aVariable->loadValue( ( theK1[ c ] * ( 35.0 / 384.0 ) 
				+ theK2[ c ] * 0.0
				+ theK3[ c ] * ( 500.0 / 1113.0 )
				+ theK4[ c ] * ( 125.0 / 192.0 )
				- theK5[ c ] * ( 2187.0 / 6784.0 )
				+ theK6[ c ] * ( 11.0 / 84.0 ) )
			      * getStepInterval()
			      + theValueBuffer[ c ] );

	// k6 * 11/84 for Yn+1
	theVelocityBuffer[ c ] += theK6[ c ] * ( 11.0 / 84.0 );
	// k6 * 187/2100 for ~Yn+1
	theErrorEstimate[ c ] += theK6[ c ] * ( 187.0 / 2100.0 );

	// clear velocity
	aVariable->setVelocity( 0.0 );
      }

    // ========= 7 ===========
    setCurrentTime( aCurrentTime + getStepInterval() );
    process();

    // evaluate error
    Real maxError( 0.0 );
	
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );

	// get k7
	theK7[ c ] = aVariable->getVelocity();

	// k7 * 1/40 for ~Yn+1
	theErrorEstimate[ c ] += theK7[ c ] * ( 1.0 / 40.0 );

	// calculate velocity for Xn+.5
	theMidVelocityBuffer[ c ] = theK1[ c ] * ( 6025192743.0 / 30085553152 )
	  + theK3[ c ] * ( 51252292925.0 / 65400821598.0 )
	  + theK4[ c ] * ( -2691868925.0 / 45128329728.0 )
	  + theK5[ c ] * ( 187940372067.0 / 1594534317056.0 )
	  + theK6[ c ] * ( -1776094331.0 / 19743644256.0 )
	  + theK7[ c ] * ( 11237099.0 / 235043384.0 );

	const Real aTolerance( eps_rel * 
			       ( a_y * fabs( theValueBuffer[ c ] ) 
				 + a_dydt * fabs( theVelocityBuffer[ c ] ) )
			       + eps_abs );

	const Real anError( fabs( ( theVelocityBuffer[ c ] 
				    - theErrorEstimate[ c ] ) / aTolerance ) );

	if( anError > maxError )
	  {
	    maxError = anError;
	  }
	
	aVariable->loadValue( theValueBuffer[ c ] );
	aVariable->setVelocity( 0.0 );
      }

    setMaxErrorRatio( maxError );
    setCurrentTime( aCurrentTime );

    if( maxError > 1.1 )
      {
	// reset the stepper current time
	reset();
	setOriginalStepInterval( 0.0 );
	theInterrupted = true;

	return false;
      }

    // set the error limit interval
    setOriginalStepInterval( getStepInterval() );
    theInterrupted = false;

    return true;
  }

  void DormandPrince547MStepper::interrupt( StepperPtr aCaller )
  {
    theInterrupted = true;
    AdaptiveDifferentialStepper::interrupt( aCaller );
  }

} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/

