//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2001 Keio University
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
// E-CELL Project, Institute for Advanced Biosciences, Keio University.
//

#include "PyUVariable.hpp"

PyUConstant::PyUConstant( const Py::Object& aPyObject )
{
  // FIXME: ugly to use _TYPE_Check()?
  if( Py::_Float_Check( *aPyObject ) )
    {
      theData = new libecs::UConstantRealData( Py::Float( aPyObject ) );
      theType = REAL;
    }
  else if( Py::_Int_Check( *aPyObject ) )
    {
      theData = new libecs::UConstantIntData( Py::Int( aPyObject ) );
      theType = INT;
    }
  else if( Py::_Long_Check( *aPyObject ) )
    {
      theData = new libecs::UConstantRealData( Py::Float( aPyObject ) );
      theType = REAL;
    }
  else // assume everything else as a string
    {
      theData = new libecs::UConstantStringData( aPyObject.as_string() );
      theType = STRING;
    }
}

const Py::Object PyUConstant::asPyObject() const
{
  switch( getType() )
    {
    case UConstant::REAL:
      return Py::Float( asReal() );
      
    case UConstant::INT:
      // FIXME: ugly... determine the type by autoconf?
      return Py::Int( static_cast<long int>( asInt() ) );
      
    case UConstant::STRING:
    case UConstant::NONE:
      return Py::String( asString() );
      
    default:
      Py::SystemError( "Unexpected error at: " + 
		       libecs::String( __PRETTY_FUNCTION__  ));
    }

}


