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
// written by Kouichi Takahashi <shafi@e-cell.org>,
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//

#include <limits>

#include "Model.hpp"
#include "System.hpp"
#include "Variable.hpp"

#include "SystemStepper.hpp"


namespace libecs
{

  LIBECS_DM_INIT_STATIC( SystemStepper, Stepper );

  ////////////////////////// Stepper

  SystemStepper::SystemStepper() 
  {
    setCurrentTime( INF );
    setMaxStepInterval( INF );
    setStepInterval( INF );
    setPriority( std::numeric_limits<Integer>::max() ); 
  }


  SystemStepper::~SystemStepper()
  {
    ; // do nothing
  }

  void SystemStepper::step()
  {
    setStepInterval( INF );
  }

  void SystemStepper::integrate( RealParam aTime )
  {
    integrateVariablesRecursively( getModel()->getRootSystem(), aTime );
    setCurrentTime( aTime );
  }

  void SystemStepper::integrateVariablesRecursively( SystemPtr const aSystem,
						     RealParam aTime )
  {
    
    FOR_ALL( VariableMap, aSystem->getVariableMap() )
      {
	VariablePtr const aVariable( i->second );
	
	if( aVariable->isIntegrationNeeded() )
	  {
	    aVariable->integrate( aTime );
	  }
      }

    FOR_ALL( SystemMap, aSystem->getSystemMap() )
      {
	SystemPtr const aSubSystem( i->second );
	integrateVariablesRecursively( aSubSystem, aTime );
      }
  }

  void SystemStepper::initialize()
  {
    // all Steppers are dependent on this SystemStepper.
    SystemStepper::updateDependentStepperVector();
  }

  void SystemStepper::updateDependentStepperVector()
  {
    theDependentStepperVector.clear();

    StepperMapCref aStepperMap( getModel()->getStepperMap() );

    FOR_ALL( StepperMap, aStepperMap )
      {
	StepperPtr aStepper( i->second );

	theDependentStepperVector.push_back( aStepper );
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

