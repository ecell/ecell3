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

#include "Primitive.hpp"

const String Primitive::PrimitiveTypeString( Type type )
{
  String aString;

  switch( type )
    {
    case ENTITY:
      aString = String( "Entity" );
      break;
    case SUBSTANCE:
      aString = String( "Substance" );
      break;
    case REACTOR:
      aString = String( "Reactor" );
      break;
    case SYSTEM:
      aString = String( "System" );
      break;
    default:
      throw InvalidPrimitiveType( __PRETTY_FUNCTION__, 
				  "can't create PrimitiveTypeString." );
    }

  return aString;
}

Primitive::Type Primitive::PrimitiveType( StringCref typestring )
{
  Type aType;

  if( typestring == "Entity" )
    {
      aType = ENTITY;
    }
  else if( typestring == "Substance" )
    {
      aType = SUBSTANCE;
    }
  else if( typestring == "Reactor" )
    {
      aType = REACTOR;
    }
  else if( typestring == "System" )
    {
      aType = SYSTEM;
    }
  else
    {
      throw InvalidPrimitiveType( __PRETTY_FUNCTION__, 
				  "can't convert typestring [" + typestring
				  + "] to PrimitiveType." );
    }

  return aType;
}



/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
