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

#include <assert.h>

#include "Exceptions.hpp"
#include "EntityType.hpp"


namespace libecs
{

  StringCref EntityType::EntityTypeStringOfEntity()
  {
    const static String aString( "Entity" );
    return aString;
  }

  StringCref EntityType::EntityTypeStringOfProcess()
  {
    const static String aString( "Process" );
    return aString;
  }
  
  StringCref EntityType::EntityTypeStringOfVariable()
  {
    const static String aString( "Variable" );
    return aString;
  }
  
  StringCref EntityType::EntityTypeStringOfSystem()
  { 
    const static String aString( "System" );
    return aString;
  }



  EntityType::EntityType( StringCref typestring )
  {
    if( typestring == EntityTypeStringOfVariable() )
      {
	theType = VARIABLE;
      }
    else if( typestring == EntityTypeStringOfProcess() )
      {
	theType = PROCESS;
      }
    else if( typestring == EntityTypeStringOfSystem() )
      {
	theType = SYSTEM;
      }
    else if( typestring == EntityTypeStringOfEntity() )
      {
	theType = ENTITY;
      }
    else
      {
	THROW_EXCEPTION( InvalidEntityType,
			 "can't convert typestring [" + typestring
			 + "] to EntityType." );
      }
  }

  EntityType::EntityType( const Int number )
    :
    theType( static_cast<const Type>( number ) )
  {
    if( number > 4 || number <= 0 )
      {
	THROW_EXCEPTION( InvalidEntityType,
			 "Invalid EntityType number" );
      }
  }

  StringCref EntityType::getString() const
  {
    switch( theType )
      {
      case VARIABLE:
	return EntityTypeStringOfVariable();
      case PROCESS:
	return EntityTypeStringOfProcess();
      case SYSTEM:
	return EntityTypeStringOfSystem();
      case ENTITY:
	return EntityTypeStringOfEntity();
      default:
	THROW_EXCEPTION( InvalidEntityType,
			 "unexpected EntityType::Type." );
      }
  }

} // namespace libecs

