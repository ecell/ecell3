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
#include "convertTo.hpp"

#include "Polymorph.hpp"

namespace libecs
{

  PolymorphStringData::PolymorphStringData( RealCref f )
    :
    theValue( convertTo<String>( f ) )
  {
    ; // do nothing
  }

  PolymorphStringData::PolymorphStringData( IntCref i )
    :
    theValue( convertTo<String>( i ) )
  {
    ; // do nothing
  }

  const Real PolymorphStringData::asReal() const
  {
    return convertTo<Real>( theValue );
  }

  const Int PolymorphStringData::asInt() const
  {
    return convertTo<Int>( theValue );
  }


  PolymorphRealData::PolymorphRealData( StringCref str )
    :
    theValue( convertTo<Real>( str ) )
  {
    ; // do nothing
  }

  const String PolymorphRealData::asString() const
  {
    return convertTo<String>( theValue );
  }

  const Int PolymorphRealData::asInt() const 
  { 
    return convertTo<Int>( theValue ); 
  }

  PolymorphIntData::PolymorphIntData( StringCref str )
    :
    theValue( convertTo<Int>( str ) )
  {
    ; // do nothing
  }

  PolymorphIntData::PolymorphIntData( RealCref f )
    :
    // FIXME: range check?
    theValue( convertTo<Int>( f ) )
  {
    ; // do nothing
  }


  const String PolymorphIntData::asString() const
  {
    return convertTo<String>( theValue );
  }

  const Real PolymorphIntData::asReal() const 
  { 
    return convertTo<Real>( theValue ); 
  }

  const Polymorph::Type Polymorph::getType() const
  {
    if( typeid( *theData) == typeid( PolymorphRealData ) )
      {
	return REAL;
      }
    else if( typeid( *theData ) == typeid( PolymorphIntData ) )
      {
	return INT;
      }
    else if( typeid( *theData ) == typeid( PolymorphStringData ) )
      {
	return STRING;
      }
    else if( typeid( *theData ) == typeid( PolymorphNoneData ) )
      {
	return NONE;
      }
    
    THROW_EXCEPTION( UnexpectedError, "NEVER_GET_HERE" );
  }


} // namespace libecs
