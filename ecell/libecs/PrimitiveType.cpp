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

#include <assert.h>

#include "Exceptions.hpp"
#include "PrimitiveType.hpp"


namespace libecs
{

  PrimitiveType::PrimitiveType( StringCref typestring )
  {
    if( typestring == PrimitiveTypeStringOfSubstance() )
      {
	theType = SUBSTANCE;
      }
    else if( typestring == PrimitiveTypeStringOfReactor() )
      {
	theType = REACTOR;
      }
    else if( typestring == PrimitiveTypeStringOfSystem() )
      {
	theType = SYSTEM;
      }
    else if( typestring == PrimitiveTypeStringOfEntity() )
      {
	theType = ENTITY;
      }
    else
      {
	throw InvalidPrimitiveType( __PRETTY_FUNCTION__, 
				    "can't convert typestring [" + typestring
				    + "] to PrimitiveType." );
      }
  }

  PrimitiveType::PrimitiveType( const Int number )
    :
    theType( static_cast<const Type>( number ) )
  {
    if( number > 4 || number <= 0 )
      {
	throw InvalidPrimitiveType( __PRETTY_FUNCTION__, 
				    "Invalid PrimitiveType number" );
      }
  }

  StringCref PrimitiveType::getString() const
  {
    switch( theType )
      {
      case SUBSTANCE:
	return PrimitiveTypeStringOfSubstance();
      case REACTOR:
	return PrimitiveTypeStringOfReactor();
      case SYSTEM:
	return PrimitiveTypeStringOfSystem();
      case ENTITY:
	return PrimitiveTypeStringOfEntity();
      default:
	throw InvalidPrimitiveType( __PRETTY_FUNCTION__, 
				    "unexpected PrimitiveType::Type." );
      }
  }

} // namespace libecs

