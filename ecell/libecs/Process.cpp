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

#include <iostream>

#include "Util.hpp"
#include "Connection.hpp"
#include "Stepper.hpp"
#include "FullID.hpp"
#include "Variable.hpp"
#include "Model.hpp"
#include "PropertySlotMaker.hpp"

#include "Process.hpp"


namespace libecs
{

  void Process::makeSlots()
  {
    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "Connection", *this, 
				      Type2Type<Polymorph>(),
				      &Process::setConnection,
				      NULLPTR ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "ConnectionList", *this, 
				      Type2Type<Polymorph>(),
				      &Process::setConnectionList,
				      &Process::getConnectionList ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "Activity", *this, 
				      Type2Type<Real>(),
				      &Process::setActivity,
				      &Process::getActivity ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "Priority", *this, 
				      Type2Type<Int>(),
				      &Process::setPriority,
				      &Process::getPriority ) );
  }

  void Process::setConnection( PolymorphCref aValue )
  {
    PolymorphVector aVector( aValue.asPolymorphVector() );
    checkSequenceSize( aVector, 3 );

    std::cerr << "Use of Process::setConnection() is deprecated. Use ConnectionList." << std::endl;

    registerConnection( aVector[0].asString(), FullID( aVector[1].asString() ), 
		      aVector[2].asInt() );
  }

  void Process::setConnectionList( PolymorphCref aValue )
  {
    const PolymorphVector aVector( aValue.asPolymorphVector() );
    for( PolymorphVectorConstIterator i( aVector.begin() );
	 i != aVector.end(); ++i )
      {
	const PolymorphVector anInnerVector( (*i).asPolymorphVector() );

	// Require ( tagname, fullid, coefficient ) 3-tuple
	if( anInnerVector.size() < 3 )
	  {
	    THROW_EXCEPTION( ValueError, "Process [" + getFullID().getString()
			     + "]: ill-formed ConnectionList given." );
	  }

	const String aConnectionName(  anInnerVector[0].asString() );
	const FullID aFullID(        anInnerVector[1].asString() );
	const Int    aCoefficient( anInnerVector[2].asInt() );

	registerConnection( aConnectionName, aFullID, aCoefficient );
      }

  }

  const Polymorph Process::getConnectionList() const
  {
    PolymorphVector aVector;
    aVector.reserve( theConnectionMap.size() );
  
    for( ConnectionMapConstIterator i( theConnectionMap.begin() );
	 i != theConnectionMap.end() ; ++i )
      {
	PolymorphVector anInnerVector;
	ConnectionCref aConnection( i->second );

	// Tagname
	anInnerVector.push_back( i->first );
	// FullID
	anInnerVector.push_back( aConnection.getVariable()->
				 getFullID().getString() );
	// Coefficient
	anInnerVector.push_back( aConnection.getCoefficient() );

	aVector.push_back( anInnerVector );
      }

    return aVector;
  }

  void Process::registerConnection( StringCref aName, FullIDCref aFullID, 
				  const Int aCoefficient )
  {
    SystemPtr aSystem( getModel()->getSystem( aFullID.getSystemPath() ) );
    VariablePtr aVariable( aSystem->getVariable( aFullID.getID() ) );

    registerConnection( aName, aVariable, aCoefficient );
  }

  Process::Process() 
    :
    theActivity( 0.0 ),
    thePriority( 0 )
  {
    makeSlots();
  }

  Process::~Process()
  {
    ; // do nothing
  }


  void Process::registerConnection( StringCref aName, VariablePtr aVariable, 
				  const Int aCoefficient )
  {
    Connection aConnection( aVariable, aCoefficient );
    theConnectionMap.insert( ConnectionMap::value_type( aName, aConnection ) );
  }


  Connection Process::getConnection( StringCref aName )
  {
    ConnectionMapConstIterator anIterator( theConnectionMap.find( aName ) );

    if( anIterator == theConnectionMap.end() )
      {
	THROW_EXCEPTION( NotFound,
			 "[" + getFullID().getString() + 
			 "]: Connection [" + aName + 
			 "] not found in this Process." );
      }

    return ( *anIterator ).second;
  }

  void Process::initialize()
  {
    ; // do nothing
  }


} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
