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

#include "Util.hpp"
#include "System.hpp"
#include "Accumulators.hpp"
#include "AccumulatorMaker.hpp"
#include "Model.hpp"
#include "EntityType.hpp"
#include "PropertySlotMaker.hpp"

#include "Substance.hpp"


namespace libecs
{

  void Substance::makeSlots()
  {
    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "Quantity",*this,
				      Type2Type<Real>(),
				      &Substance::setQuantity,
				      &Substance::getQuantity ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "Velocity",*this,
				      Type2Type<Real>(),
				      &Substance::addVelocity,
				      &Substance::getVelocity ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "Fixed",*this,
				      Type2Type<Int>(),
				      &Substance::setFixed,
				      &Substance::getFixed ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "Concentration", *this,
				      Type2Type<Real>(),
				      NULLPTR,
				      &Substance::getConcentration ) );

  }


  Substance::Substance()
    : 
    theQuantity( 0.0 ),  
    theVelocity( 0.0 ),
    theFixed( false )
  {
    makeSlots();
  } 

  Substance::~Substance()
  {
    ; // do nothing
  }


  void Substance::initialize()
  {

  }

  void Substance::setFixed( IntCref aValue )
  { 
    theFixed = static_cast<bool>( aValue );
  }


  const Int Substance::getFixed() const                         
  { 
    return theFixed;
  }




    const String SRMSubstance::SYSTEM_DEFAULT_ACCUMULATOR_NAME = "ReserveAccumulator";

  //  const String SRMSubstance::SYSTEM_DEFAULT_ACCUMULATOR_NAME = "SimpleAccumulator";


  SRMSubstance::SRMSubstance()
    : 
    theAccumulator( NULLPTR ),
    theFraction( 0 )
  {
    makeSlots();
    // FIXME: use AccumulatorMaker
    setAccumulator( new ReserveAccumulator );
  } 

  SRMSubstance::~SRMSubstance()
  {
    delete theAccumulator;
  }


  const String SRMSubstance::getAccumulatorClass() const
  {
    return theAccumulator->getClassName();
  }


  void SRMSubstance::setAccumulatorClass( StringCref anAccumulatorClassname )
  {
    AccumulatorPtr aAccumulatorPtr( getModel()->getAccumulatorMaker()
				    .make( anAccumulatorClassname ) );
    setAccumulator( aAccumulatorPtr );
  }

  void SRMSubstance::setAccumulator( AccumulatorPtr anAccumulator )
  {
    if( theAccumulator != NULLPTR )
      {
	delete theAccumulator;
      }

    theAccumulator = anAccumulator;
    theAccumulator->setOwner( this );
    //    theAccumulator->update();
  }



  void SRMSubstance::initialize()
  {
    // if the accumulator is not set, use user default
    if( theAccumulator == NULLPTR )
      {
	setAccumulatorClass( SYSTEM_DEFAULT_ACCUMULATOR_NAME );
      }

    theAccumulator->update();
  }


  const Real SRMSubstance::saveQuantity()
  {
    return theAccumulator->save();
  }



  void SRMSubstance::loadQuantity( RealCref aQuantity )
  {
    Substance::loadQuantity( aQuantity );
    theAccumulator->update();
  }


  void SRMSubstance::integrate()
  { 
    if( isFixed() == false ) 
      {
	Substance::integrate();

	theAccumulator->accumulate();
  
	if( theQuantity < 0.0 ) 
	  {
	    theQuantity = 0.0;
	    //FIXME:       throw LTZ();
	  }
      }
  }

  void SRMSubstance::makeSlots()
  {
    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "AccumulatorClass",*this,
				      Type2Type<String>(),
				      &SRMSubstance::setAccumulatorClass,
				      &SRMSubstance::getAccumulatorClass ) );

  }


} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
