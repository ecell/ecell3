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

#include "Util.hpp"
#include "Reactant.hpp"
#include "RootSystem.hpp"
#include "Stepper.hpp"
#include "FullID.hpp"
#include "Substance.hpp"

#include "Reactor.hpp"


namespace libecs
{

  void Reactor::makeSlots()
  {
    //FIXME: get methods
    createPropertySlot( "Reactant", *this, 
			&Reactor::setReactant,
			NULLPTR );

    createPropertySlot( "ReactantList", *this, 
			NULLPTR,
			&Reactor::getReactantList);

    createPropertySlot( "InitialActivity", *this, 
			&Reactor::setInitialActivity,
			&Reactor::getInitialActivity );
  }

  void Reactor::setReactant( UVariableVectorRCPtrCref aMessage )
  {
    //FIXME: range check
    registerReactant( (*aMessage)[0].asString(), 
		      FullID( (*aMessage)[1].asString() ), 
		      (*aMessage)[2].asInt() );
  }

  const UVariableVectorRCPtr Reactor::getReactantList() const
  {
    UVariableVectorRCPtr aVectorPtr( new UVariableVector );
    aVectorPtr->reserve( theReactantMap.size() );
  
    for( ReactantMapConstIterator i( theReactantMap.begin() );
	 i != theReactantMap.end() ; ++i )
      {
	aVectorPtr->push_back( i->second.getSubstance()->getFullID().getString() );
      }

    return aVectorPtr;
  }

  void Reactor::registerReactant( StringCref aName, FullIDCref aFullID, 
				  const Int aStoichiometry )
  {
    SystemPtr aRootSystem( getRootSystem() );
    SystemPtr aSystem( aRootSystem->getSystem( aFullID.getSystemPath() ) );
    SubstancePtr aSubstance( aSystem->getSubstance( aFullID.getID() ) );

    registerReactant( aName, aSubstance, aStoichiometry );
  }


  void Reactor::setInitialActivity( RealCref anActivity )
  {
    theInitialActivity = anActivity;

    theActivity = theInitialActivity * 
      getSuperSystem()->getStepper()->getStepInterval();
  }

  Reactor::Reactor() 
    :
    theInitialActivity( 0 ),
    theActivity( 0 )
  {
    makeSlots();
  }

  Reactor::~Reactor()
  {
    // delete all Reactants
    //    for( ReactantMapConstIterator i( theReactantMap.begin() );
    //	 i != theReactantMap.end() ; ++i )
    //      {
    //	delete i->second;
    //      }
  }


  void Reactor::registerReactant( StringCref aName, SubstancePtr aSubstance, 
				  const Int aStoichiometry )
  {
    Reactant aReactant( aSubstance, aStoichiometry );
    theReactantMap.insert( ReactantMap::value_type( aName, aReactant ) );
  }


  Reactant Reactor::getReactant( StringCref aName )
  {
    ReactantMapConstIterator anIterator( theReactantMap.find( aName ) );

    if( anIterator == theReactantMap.end() )
      {
	throw NotFound( __PRETTY_FUNCTION__, 
			"[" + getFullID().getString() + 
			"]: Reactant [" + aName + 
			"] not found in this Reactor." );
      }

    return ( *anIterator ).second;
  }

  const Int Reactor::getNumberOfReactants() const
  {
    return theReactantMap.size();
  }

  void Reactor::initialize()
  {
    ; // do nothing
  }

  PropertySlotPtr 
  Reactor::getPropertySlotOfReactant( StringCref aReactantName,
				      StringCref aPropertyName )
  {
    return getReactant( aReactantName ).
      getSubstance()->getPropertySlot( aPropertyName, this );
  }




} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
