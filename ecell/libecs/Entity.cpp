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

#include "Entity.hpp"
#include "System.hpp"
#include "FullID.hpp"
#include "Stepper.hpp"

namespace libecs
{

  Entity::Entity()
    : 
    theSuperSystem( NULLPTR ),
    theID( "" ),
    theName( "" ) 
  {
    makeSlots();
  }


  Entity::~Entity()
  {
    ; // do nothing
  }

  void Entity::makeSlots()
  {
    makePropertySlot( "ClassName", Entity, *this, NULLPTR, &Entity::getClassName );
    makePropertySlot( "ID", Entity, *this, NULLPTR, &Entity::getID );
    makePropertySlot( "FullID", Entity, *this, NULLPTR, &Entity::getFullID );
    
    makePropertySlot( "Name", Entity, *this, NULLPTR, &Entity::getName );
    makePropertySlot( "Activity", Entity, *this, NULLPTR, &Entity::getActivity );
    makePropertySlot( "ActivityPerSecond", Entity, *this, NULLPTR, 
		     &Entity::getActivityPerSecond );
  }

  const Message Entity::getClassName( StringCref keyword )
  {
    return Message( keyword, UConstant( getClassName() ) );
  }

  const Message Entity::getID( StringCref keyword )
  {
    return Message( keyword, UConstant( getID() ) );
  }

  const Message Entity::getFullID( StringCref keyword )
  {

    return Message( keyword, UConstant( getFullID().getString() ) );


#if 0
    FullID aFullID = getFullID();
    UConstantVector aVector( 3 );

    aVector[0] = static_cast<Int>( aFullID.getPrimitiveType().getType() );
    aVector[1] = aFullID.getSystemPath().getString();
    aVector[2] = aFullID.getID();

    return Message( keyword, aVector );
#endif /* 0 */

  }

  const Message Entity::getName( StringCref keyword )
  {
    return Message( keyword, UConstant( getName() ) );
  }

  const Message Entity::getActivity( StringCref keyword )
  {
    return Message( keyword, UConstant( getActivity() ) );
  }

  const Message Entity::getActivityPerSecond( StringCref keyword )
  {
    return Message( keyword, UConstant( getActivityPerSecond() ) );
  }

  Real Entity::getActivity() 
  {
    return 0;
  }

  Real Entity::getActivityPerSecond() 
  {
    return getActivity() * getSuperSystem()->getStepper()->getStepsPerSecond();
  }

  const FullID Entity::getFullID() const
  {
    return FullID( getPrimitiveType(), getSystemPath(), getID() );
  }

  const SystemPath Entity::getSystemPath() const
  {
    SystemPtr aSystemPtr( getSuperSystem() );
    SystemPath aSystemPath( aSystemPtr->getSystemPath() );
    aSystemPath.push_back( aSystemPtr->getID() );
    return aSystemPath;
  }


} // namespace libecs

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
