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

#include <algorithm>

#include "System.hpp"
#include "Reactor.hpp"
#include "Model.hpp"
#include "SubstanceMaker.hpp"
#include "ReactorMaker.hpp"
#include "Substance.hpp"
#include "Stepper.hpp"
#include "StepperMaker.hpp"
#include "FullID.hpp"
#include "SystemMaker.hpp"
#include "PropertyInterface.hpp"

namespace libecs
{


  /////////////////////// System

  void System::makeSlots()
  {
    createPropertySlot( "SystemList", *this,
			NULLPTR, &System::getSystemList );
    createPropertySlot( "SubstanceList", *this,
			NULLPTR, &System::getSubstanceList );
    createPropertySlot( "ReactorList", *this,
			NULLPTR, &System::getReactorList );

    createPropertySlot( "StepperID", *this,
			&System::setStepperID, &System::getStepperID );

    createPropertySlot( "Volume", *this,
			&System::setVolume, &System::getVolume );

    createPropertySlot( "StepInterval", *this,
			&System::setStepInterval, &System::getStepInterval );
  }


  // Message slots

  const UVariableVectorRCPtr System::getSystemList() const
  {
    UVariableVectorRCPtr aVectorPtr( new UVariableVector );
    aVectorPtr->reserve( theSystemMap.size() );

    for( SystemMapConstIterator i = getSystemMap().begin() ;
	 i != getSystemMap().end() ; ++i )
      {
	aVectorPtr->push_back( i->second->getID() );
      }

    return aVectorPtr;
  }

  const UVariableVectorRCPtr System::getSubstanceList() const
  {
    UVariableVectorRCPtr aVectorPtr( new UVariableVector );
    aVectorPtr->reserve( theSubstanceMap.size() );

    for( SubstanceMapConstIterator i( getSubstanceMap().begin() );
	 i != getSubstanceMap().end() ; ++i )
      {
	aVectorPtr->push_back( i->second->getID() );
      }

    return aVectorPtr;
  }

  const UVariableVectorRCPtr System::getReactorList() const
  {
    UVariableVectorRCPtr aVectorPtr( new UVariableVector );
    aVectorPtr->reserve( theReactorMap.size() );

    for( ReactorMapConstIterator i( getReactorMap().begin() );
	 i != getReactorMap().end() ; ++i )
      {
	aVectorPtr->push_back( i->second->getID() );
      }

    return aVectorPtr;
  }


  void System::setStepperID( UVariableVectorRCPtrCref aMessage )
  {
    checkSequenceSize( *aMessage, 1 );

    setStepperID( (*aMessage)[0].asString() );
  }

  const UVariableVectorRCPtr System::getStepperID() const
  {
    UVariableVectorRCPtr aVector( new UVariableVector );
    //    aVector->push_back( UVariable( getModel()->
    //				   getStepper()->getClassName() ) );

    aVector->push_back( UVariable( "NOT IMPLEMENTED YET" ) );
    return aVector;
  }

  System::System()
    :
    theVolume( 1 ),
    theConcentrationFactor( 0 ),
    theStepper( NULLPTR ),
    theEntityListChanged( false )
  {
    makeSlots();
    theFirstRegularReactorIterator = getReactorMap().begin();

    updateConcentrationFactor();
  }

  System::~System()
  {
    getStepper()->disconnectSystem( this );
    
    for( ReactorMapIterator i( theReactorMap.begin() );
	 i != theReactorMap.end() ; ++i )
      {
	delete i->second;
      }

    for( SubstanceMapIterator i( theSubstanceMap.begin() );
	 i != theSubstanceMap.end() ; ++i )
      {
	delete i->second;
      }

    for( SystemMapIterator i( theSystemMap.begin() );
	 i != theSystemMap.end() ; ++i )
      {
	delete i->second;
      }
  }

  void System::setStepperID( StringCref anID )
  {
    theStepper = getModel()->getStepper( anID );
    theStepper->connectSystem( this );
  }

  void System::setStepInterval( RealCref aStepInterval )
  {
    theStepper->setStepInterval( aStepInterval );
  }

  const Real System::getStepInterval() const
  {
    return theStepper->getStepInterval();
  }

  const Real System::getStepsPerSecond() const
  {
    return theStepper->getStepsPerSecond();
  }

  void System::initialize()
  {
    if( theStepper == NULLPTR )
      {
	THROW_EXCEPTION( InitializationFailed,
			 getFullID().getString() + ": Stepper not set." );
      }

    //
    // Substance::initialize()
    //
    for( SubstanceMapConstIterator i( getSubstanceMap().begin() );
	 i != getSubstanceMap().end() ; ++i )
      {
	i->second->initialize();
      }

    //
    // Reactor::initialize()
    //
    for( ReactorMapConstIterator i( getReactorMap().begin() );
	 i != getReactorMap().end() ; ++i )
      {
	i->second->initialize();
      }

    theFirstRegularReactorIterator = find_if( getReactorMap().begin(),
					      getReactorMap().end(),
					      isRegularReactorItem() );

    // do not need to call subsystems' initialize()


    updateConcentrationFactor();
  }

  void System::registerReactor( ReactorPtr aReactor )
  {
    const String anID( aReactor->getID() );

    if( getReactorMap().find( anID ) != getReactorMap().end() )
      {
	delete aReactor;

	THROW_EXCEPTION( AlreadyExist, 
			 "[" + getFullID().getString() + 
			 "]: Reactor [" + anID + "] already exist." );
      }

    theReactorMap[ anID ] = aReactor;
    aReactor->setSuperSystem( this );
    aReactor->setModel( getModel() );

    notifyChangeOfEntityList();
  }

  ReactorPtr System::getReactor( StringCref anID ) 
  {
    ReactorMapConstIterator i( getReactorMap().find( anID ) );

    if( i == getReactorMap().end() )
      {
	THROW_EXCEPTION( NotFound, 
			 "[" + getFullID().getString() + 
			"]: Reactor [" + anID + 
			 "] not found in this System." );
      }

    return i->second;
  }

  void System::registerSubstance( SubstancePtr aSubstance )
  {
    const String anID( aSubstance->getID() );

    if( getSubstanceMap().find( anID ) != getSubstanceMap().end() )
      {
	delete aSubstance;

	THROW_EXCEPTION( AlreadyExist, 
			 "[" + getFullID().getString() + 
			 "]: Substance [" + anID + "] already exist." );
      }

    theSubstanceMap[ anID ] = aSubstance;
    aSubstance->setSuperSystem( this );
    aSubstance->setModel( getModel() );

    notifyChangeOfEntityList();
  }

  SubstancePtr System::getSubstance( StringCref anID ) 
  {
    SubstanceMapConstIterator i( getSubstanceMap().find( anID ) );
    if( i == getSubstanceMap().end() )
      {
	THROW_EXCEPTION( NotFound,
			 "[" + getFullID().getString() + 
			 "]: Substance [" + anID + 
			 "] not found in this System.");
      }

    return i->second;
  }


  void System::registerSystem( SystemPtr aSystem )
  {
    const String anID( aSystem->getID() );

    if( getSystemMap().find( anID ) != getSystemMap().end() )
      {
	delete aSystem;

	THROW_EXCEPTION( AlreadyExist, 
			 "[" + getFullID().getString() + 
			 "]: System [" + anID + "] already exist." );
      }

    theSystemMap[ anID ] = aSystem;
    aSystem->setSuperSystem( this );
    aSystem->setModel( getModel() );

    notifyChangeOfEntityList();
  }

  SystemPtr System::getSystem( StringCref anID ) 
  {
    SystemMapConstIterator i( getSystemMap().find( anID ) );
    if( i == getSystemMap().end() )
      {
	THROW_EXCEPTION( NotFound,
			 "[" + getFullID().getString() + 
			"]: System [" + anID + "] not found in this System." );
      }
    return i->second;
  }




  const Real System::getActivityPerSecond() const
  {
    return getActivity() * getStepsPerSecond();
  }

  void System::notifyChangeOfEntityList()
  {
    //    getStepper()->getMasterStepper()->setEntityListChanged();
  }

  const SystemPath System::getSystemPath() const
  {
    if( isRootSystem() )
      {
	return SystemPath();
      }
    else
      {
	return Entity::getSystemPath();
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
