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

#include <typeinfo>

#include "Util.hpp"
#include "Exceptions.hpp"

#include "Polymorph.hpp"

namespace libecs
{
  PolymorphData::~PolymorphData()
  {
    ; // do nothing
  }

  PolymorphNoneData::~PolymorphNoneData()
  {
    ; // do nothing
  }

  const PolymorphVector PolymorphNoneData::asPolymorphVector() const
  { 
    return PolymorphVector(); 
  }


  const Polymorph::Type Polymorph::getType() const
  {
    if( typeid( *theData) == typeid( ConcretePolymorphData<Real> ) )
      {
	return REAL;
      }
    else if( typeid( *theData) == typeid( ConcretePolymorphData<Int> ) )
      {
	return INT;
      }
    else if( typeid( *theData) == typeid( ConcretePolymorphData<String> ) )
      {
	return STRING;
      }
    else if( typeid( *theData) == 
	     typeid( ConcretePolymorphData<PolymorphVector> ) )
      {
	return POLYMORPH_VECTOR;
      }
    else if( typeid( *theData ) == typeid( PolymorphNoneData ) )
      {
	return NONE;
      }
    
    NEVER_GET_HERE;
  }


  void Polymorph::changeType( const Type aType )
  {
    PolymorphDataPtr aPolymorphDataPtr( NULLPTR );

    switch( aType )
      {
      case REAL:
	aPolymorphDataPtr = 
	  new ConcretePolymorphData<Real>( theData->asReal() );
	break;
      case INT:
	aPolymorphDataPtr = 
	  new ConcretePolymorphData<Int>( theData->asInt() );
	break;
      case STRING:
	aPolymorphDataPtr = 
	  new ConcretePolymorphData<Int>( theData->asString() );
	break;
      case POLYMORPH_VECTOR:
	aPolymorphDataPtr = 
	  new ConcretePolymorphData<Int>( theData->asPolymorphVector() );
	break;
      case NONE:
	aPolymorphDataPtr = new PolymorphNoneData();
	break;
      default:
	NEVER_GET_HERE;
      }

    delete theData;
    theData = aPolymorphDataPtr;
  }


} // namespace libecs
