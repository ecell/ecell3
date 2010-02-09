//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2009 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell System is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell System is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell System -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Kazunari Kaizu <kaizu@sfc.keio.ac.jp>,
// E-Cell Project.
//

#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include "BooleanStepper.hpp"

//////////////////// BooleanStepper

LIBECS_DM_INIT( BooleanStepper, Stepper );

void BooleanStepper::initialize()
{
  Stepper::initialize();

  theBooleanProcessVector.clear();

  try
  {
    std::transform( theProcessVector.begin(), theProcessVector.end(),
		    std::back_inserter( theBooleanProcessVector ),
		    DynamicCaster<BooleanProcessPtr, ProcessPtr>() );
  }
  catch ( const libecs::TypeError& )
  {
    THROW_EXCEPTION( InitializationFailed,
		     getClassNameString() +
		     ": Only BooleanProcesses are allowed to exist "
		     "in this Stepper." );
  }
}

void BooleanStepper::step()
{
  const VariableVector::size_type 
    aReadOnlyVariableOffset( getReadOnlyVariableOffset() );

  FOR_ALL( BooleanProcessVector, theBooleanProcessVector )
    {
      (*i)->evaluate();
    }

  for ( VariableVector::size_type c( 0 ); c < aReadOnlyVariableOffset; ++c )
    {
      VariablePtr const aVariable( theVariableVector[ c ] );
      aVariable->loadValue( 0.0 );
    }

  FOR_ALL( BooleanProcessVector, theBooleanProcessVector )
    {
      const Real anActivity( (*i)->getActivity() );
      (*i)->addValue( anActivity );
    }

  for ( VariableVector::size_type c( 0 ); c < aReadOnlyVariableOffset; ++c )
    {
      VariablePtr const aVariable( theVariableVector[ c ] );
      const Real aValue( aVariable->getValue() );

      if ( aValue > 0 )
	{
	  aVariable->loadValue( 1.0 );
	}
      else if ( aValue < 0 )
	{
	  aVariable->loadValue( 0.0 );
	}
      else
	{
	  ; // do nothing if aValue == 0
	}
    }
}


/*
  Do not modify
  $Author: moriyoshi $
  $Revision: 3386 $
  $Date: 2009-02-05 15:01:04 +0900 (木, 05  2月 2009) $
  $Locker$
*/

