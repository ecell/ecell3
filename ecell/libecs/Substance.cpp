//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2000 Keio University
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

#include "Substance.hpp"
#include "System.hpp"
#include "RootSystem.hpp"
#include "Accumulators.hpp"
#include "AccumulatorMaker.hpp"
#include "PrimitiveType.hpp"
#include "Util.hpp"


namespace libecs
{

  const String Substance::SYSTEM_DEFAULT_ACCUMULATOR_NAME = "ReserveAccumulator";

  String Substance::USER_DEFAULT_ACCUMULATOR_NAME 
  = Substance::SYSTEM_DEFAULT_ACCUMULATOR_NAME;

  void Substance::makeSlots()
  {
    createPropertySlot( "Quantity",*this,&Substance::setQuantity,
			&Substance::getQuantity );
    createPropertySlot( "Concentration",*this,NULLPTR,
			&Substance::getConcentration );
    createPropertySlot( "Velocity",*this,&Substance::addVelocity,
			&Substance::getVelocity );
    createPropertySlot( "AccumulatorClass",*this,
			&Substance::setAccumulatorClass,
			&Substance::getAccumulatorClass );
  }

  void Substance::setAccumulatorClass( UVariableVectorRCPtrCref aMessage )
  {
    // FIXME: range check

    setAccumulator( (*aMessage)[0].asString() );
  }

  const UVariableVectorRCPtr Substance::getAccumulatorClass() const
  {
    UVariableVectorRCPtr aUVariableVector( new UVariableVector );
    if( theAccumulator ) 
      {
	aUVariableVector->
	  push_back( UVariable( theAccumulator->getClassName() ) );
      }

    return aUVariableVector;
  }

  Substance::Substance()
    : 
    theAccumulator( NULLPTR ),
    theIntegrator( NULLPTR ),
    theQuantity( 0 ),  
    theFraction( 0 ),
    theVelocity( 0 ),
    theFixed( false ) ,
    theConcentration( -1 )
  {
    makeSlots();
    // FIXME: use AccumulatorMaker
    setAccumulator( new ReserveAccumulator );
  } 

  Substance::~Substance()
  {
    delete theIntegrator;
    delete theAccumulator;
  }

  void Substance::setAccumulator( StringCref anAccumulatorClassname )
  {
    try {
      AccumulatorPtr aAccumulatorPtr( getRootSystem()->getAccumulatorMaker()
				      .make( anAccumulatorClassname ) );
      setAccumulator( aAccumulatorPtr );

      if( anAccumulatorClassname != userDefaultAccumulatorName() )
	{
	  //FIXME:    *theMessageWindow << "[" << fqpi() 
	  //FIXME: << "]: accumulator is changed to: " << classname << ".\n";
	}
    }
    catch( Exception& e )
      {
	//FIXME:     *theMessageWindow << "[" << fqpi() << "]:\n" << e.message();
	// warn if theAccumulator is already set
	if( theAccumulator != NULLPTR )   
	  {
	    //FIXME: *theMessageWindow << "[" << fqpi() << 
	    //FIXME: "]: falling back to :" << theAccumulator->className() 
	    //FIXME: << ".\n";
	  }
      }
  }

  void Substance::setAccumulator( AccumulatorPtr anAccumulator )
  {
    if( theAccumulator != NULLPTR )
      {
	delete theAccumulator;
      }

    theAccumulator = anAccumulator;
    theAccumulator->setOwner( this );
    theAccumulator->update();
  }

  void Substance::initialize()
  {
    if( theAccumulator == NULLPTR )
      {
	setAccumulator( USER_DEFAULT_ACCUMULATOR_NAME );
      }

    // if the user default is invalid fall back to the system default.
    if( ! theAccumulator )  
      {               
	//FIXME:      *theMessageWindow << "Substance: " 
	//FIXME:	<< "falling back to the system default accumulator: " 
	//FIXME:	  << SYSTEM_DEFAULT_ACCUMULATOR_NAME  << ".\n";
	setUserDefaultAccumulatorName( SYSTEM_DEFAULT_ACCUMULATOR_NAME );
	setAccumulator( USER_DEFAULT_ACCUMULATOR_NAME );
      }
  }

  const Real Substance::saveQuantity()
  {
    return theAccumulator->save();
  }

  void Substance::loadQuantity( RealCref aQuantity )
  {
    mySetQuantity( aQuantity );
    theAccumulator->update();
  }

  Real Substance::getActivity()
  {
    return getVelocity();
  }

  void Substance::calculateConcentration() const
  {
    theConcentration = theQuantity / ( getSuperSystem()->getVolume() * N_A ); 
  }


  //FIXME: the following methods should be inlined

  void Substance::integrate()
  { 
    theConcentration = -1;

    if( ! isFixed() ) 
      {
	theIntegrator->integrate();

	theAccumulator->accumulate();
  
	if( theQuantity < 0 ) 
	  {
	    theQuantity = 0;
	    //FIXME:       throw LTZ();
	  }
      }
  }

} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
