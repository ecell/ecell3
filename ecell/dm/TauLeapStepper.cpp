//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2004 Keio University
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
// written by Tomoya Kitayama <tomo@e-cell.org>, 
// E-Cell Project.
//

#include <gsl/gsl_randist.h>

#include "TauLeapStepper.hpp"
 
LIBECS_DM_INIT( TauLeapStepper, Stepper );

void TauLeapStepper::initialize()
{
  DifferentialStepper::initialize();
      
  theGillespieProcessVector.clear();

  try
    {
      std::transform( theProcessVector.begin(), theProcessVector.end(),
		      std::back_inserter( theGillespieProcessVector ),
		      DynamicCaster<GillespieProcessPtr,ProcessPtr>() );
    }
  catch( const libecs::TypeError& )
    {
      THROW_EXCEPTION( InitializationFailed,
		       getClassNameString() +
		       ": Only GillespieProcesses are allowed to exist "
		       "in this Stepper." );
    }
}

void TauLeapStepper::step()
{      
  clearVariables();
      
  calculateTau();

  initializeStepInterval( getTau() );
  
  FOR_ALL( GillespieProcessVector, theGillespieProcessVector )
    {
      (*i)->setActivity( gsl_ran_poisson( getRng(), (*i)->getPropensity() ) );
    }

  setVariableVelocity( theTaylorSeries[ 0 ] );
}
