//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#include <typeinfo>

#include "Util.hpp"
#include "Exceptions.hpp"

#include "Polymorph.hpp"

namespace libecs
{

  PolymorphValue::~PolymorphValue()
  {
    ; // do nothing
  }

  PolymorphNoneValue::~PolymorphNoneValue()
  {
    ; // do nothing
  }

  const PolymorphVector PolymorphNoneValue::asPolymorphVector() const
  { 
    return PolymorphVector(); 
  }

  const String PolymorphNoneValue::asString() const
  { 
    static String aNoneString;
    return aNoneString;
  }

  const Polymorph::Type Polymorph::getType() const
  {
    if( typeid( *theValue) == typeid( ConcretePolymorphValue<Real> ) )
      {
	return REAL;
      }
    else if( typeid( *theValue) == typeid( ConcretePolymorphValue<Integer> ) )
      {
	return INTEGER;
      }
    else if( typeid( *theValue) == typeid( ConcretePolymorphValue<String> ) )
      {
	return STRING;
      }
    else if( typeid( *theValue) == 
	     typeid( ConcretePolymorphValue<PolymorphVector> ) )
      {
	return POLYMORPH_VECTOR;
      }
    else if( typeid( *theValue ) == typeid( PolymorphNoneValue ) )
      {
	return NONE;
      }
    
    NEVER_GET_HERE;
  }


  void Polymorph::changeType( const Type aType )
  {
    PolymorphValuePtr aPolymorphValuePtr( NULLPTR );

    switch( aType )
      {
      case REAL:
	aPolymorphValuePtr = 
	  new ConcretePolymorphValue<Real>( theValue->asReal() );
	break;
      case INTEGER:
	aPolymorphValuePtr = 
	  new ConcretePolymorphValue<Integer>( theValue->asInteger() );
	break;
      case STRING:
	aPolymorphValuePtr = 
	  new ConcretePolymorphValue<String>( theValue->asString() );
	break;
      case POLYMORPH_VECTOR:
	aPolymorphValuePtr = 
	  new ConcretePolymorphValue<PolymorphVector>
	  ( theValue->asPolymorphVector() );
	break;
      case NONE:
	aPolymorphValuePtr = new PolymorphNoneValue();
	break;
      default:
	NEVER_GET_HERE;
      }

    delete theValue;
    theValue = aPolymorphValuePtr;
  }


} // namespace libecs
