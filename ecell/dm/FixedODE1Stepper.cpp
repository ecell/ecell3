//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2002 Keio University
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

#include "Variable.hpp"

#include "FixedODE1Stepper.hpp"

LIBECS_DM_INIT( FixedODE1Stepper, Stepper );

FixedODE1Stepper::FixedODE1Stepper()
{
  ; // do nothing
}
	    
FixedODE1Stepper::~FixedODE1Stepper()
{
  ; // do nothing
}

void FixedODE1Stepper::step()
{
  const VariableVector::size_type aSize( getReadOnlyVariableOffset() );

  clearVariables();

  setStepInterval( getNextStepInterval() );

  fireProcesses();
  setVariableVelocity( theTaylorSeries[ 0 ] );

  /**
  for( VariableVector::size_type c( 0 ); c < aSize; ++c )
    {
      VariablePtr const aVariable( theVariableVector[ c ] );
      //      theTaylorSeries[ 0 ][ c ] = aVariable->getVelocity();
    }
  */

  /**
     avoid negative value

     FOR_ALL( VariableVector, theVariableVector )
     {
     while ( (*i)->checkRange( getStepInterval() ) == false )
     {
     //FIXME:
     setStepInterval( getStepInterval() * 0.5 );
     }
     }
  */

  /**
     if ( getStepInterval() < getTolerableStepInterval() )
     {
     setNextStepInterval( getStepInterval() * 2.0 );
     }
     else 
     {
     setNextStepInterval( getStepInterval() );
     }
  */

  setNextStepInterval( getTolerableStepInterval() );
}

