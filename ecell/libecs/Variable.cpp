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

#include "Util.hpp"
#include "System.hpp"
#include "FullID.hpp"
#include "Model.hpp"
#include "EntityType.hpp"
#include "PropertySlotMaker.hpp"

#include "Variable.hpp"


namespace libecs
{

  LIBECS_DM_INIT_STATIC( Variable, Variable );
  LIBECS_DM_INIT_STATIC( PositiveVariable, Variable );


  Variable::Variable()
    : 
    theValue( 0.0 ),  
    theVelocity( 0.0 ),
    theTotalVelocity( 0.0 ),
    theLastTime( 0.0 ),
    theFixed( false )
  {
    ; // do nothing
  } 


  Variable::~Variable()
  {
    clearVariableProxyVector();
  }


  void Variable::initialize()
  {
    clearVariableProxyVector();
  }

  void Variable::clearVariableProxyVector()
  {
    for( VariableProxyVectorIterator i( theVariableProxyVector.begin() );
	   i != theVariableProxyVector.end(); ++i )
      {
	delete (*i);
      }

    theVariableProxyVector.clear();
  }


  void Variable::registerProxy( VariableProxyPtr const anVariableProxyPtr )
  {
    theVariableProxyVector.push_back( anVariableProxyPtr );
  }

  //  void Variable::removeProxy( VariableProxyPtr const anVariableProxyPtr )
  //  {
  //    theVariableProxyVector.erase( std::remove( theVariableProxyVector.begin(),
  //					       theVariableProxyVector.end(),
  //					       anVariableProxyPtr ) );
  //  }


  ///////////////////////// PositiveVariable

  SET_METHOD_DEF( Real, Value, PositiveVariable )
  {
    if( value >= 0 )
      {
	Variable::setValue( value );
      }
    else
      {
	THROW_EXCEPTION( RangeError, "PositiveVariable [" + 
			 getFullID().getString() + 
			 "]: attempt to set a negative value." );
      }
  }


  void PositiveVariable::integrate( const Real aTime )
  {
    Variable::integrate( aTime );

    //    
    // Check if the value is in positive range.
    // | value | < epsilon is rounded to zero.
    //
    const Real anEpsilon( std::numeric_limits<Real>::epsilon() );
    const Real aValue( getValue() );
    if( aValue < anEpsilon )
      {
	if( aValue > - anEpsilon )
	  {
	    loadValue( 0.0 );
	  }
	else
	  {
	    THROW_EXCEPTION( RangeError, "PositiveVariable [" + 
			     getFullID().getString() + 
			     "]: negative value occured in integrate()." );
	  }
      }
  }


} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
