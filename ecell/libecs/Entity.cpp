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
#include "FQPI.hpp"
#include "Stepper.hpp"


Entity::Entity()
  : 
  theSuperSystem( NULL ),
  theId( "" ),
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
  MessageSlot( "ClassName", Entity, *this, NULL, &Entity::getClassName );
  MessageSlot( "Id", Entity, *this, NULL, &Entity::getId );
  MessageSlot( "SystemPath", Entity, *this, NULL, &Entity::getSystemPath );
  MessageSlot( "Name", Entity, *this, NULL, &Entity::getName );
  MessageSlot( "Activity", Entity, *this, NULL, &Entity::getActivity );
  MessageSlot( "ActivityPerSecond", Entity, *this, NULL, 
	       &Entity::getActivityPerSecond );
}

const Message Entity::getClassName( StringCref keyword )
{
  return Message( keyword, UniversalVariable( getClassName() ) );
}

const Message Entity::getId( StringCref keyword )
{
  return Message( keyword, UniversalVariable( getId() ) );
}

const Message Entity::getSystemPath( StringCref keyword )
{
  return Message( keyword, UniversalVariable( getSystemPath() ) );
}

const Message Entity::getName( StringCref keyword )
{
  return Message( keyword, UniversalVariable( getName() ) );
}

const Message Entity::getActivity( StringCref keyword )
{
  return Message( keyword, UniversalVariable( getActivity() ) );
}

const Message Entity::getActivityPerSecond( StringCref keyword )
{
  return Message( keyword, UniversalVariable( getActivityPerSecond() ) );
}

Float Entity::getActivity() 
{
  return 0;
}

Float Entity::getActivityPerSecond() 
{
  return ( getActivity()  / getSuperSystem()->getStepper()->getDeltaT() );
}

const String Entity::getFqid() const
{
  String aFqid = getSystemPath();
  if( aFqid != "" )
    {
      aFqid += ":";
    }
  aFqid += getId();

  return aFqid;
}

const String Entity::getFqpi() const
{
  //FIXME: slow? use virtual methods
  return PrimitiveTypeStringOf( *this ) + ":" + getFqid();
}

const String Entity::getSystemPath() const
{
  if( getSuperSystem() == NULLPTR )
    {
      return "";
    }

  String aSystemPath = getSuperSystem()->getSystemPath(); 

  if( aSystemPath != "" )
    {
      if( aSystemPath != "/" )
	{
	  aSystemPath += SystemPath::DELIMITER;
	}
    }

  //FIXME: suspicious
  aSystemPath += getSuperSystem()->getId();

  return aSystemPath;
}

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
