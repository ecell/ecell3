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
// written by Tomoya Kitayama <tomo@e-cell.org>,
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//

#include "Model.hpp"
#include "Variable.hpp"

#include "FluxDistributionStepper.hpp"

LIBECS_DM_INIT( FluxDistributionStepper, Stepper );


FluxDistributionStepper::FluxDistributionStepper()
{
  // gcc3 doesn't currently support numeric_limits::infinity.
  // using max() instead.
  const Real anInfinity( std::numeric_limits<Real>::max() *
			 std::numeric_limits<Real>::max() );

  initializeStepInterval( anInfinity );

}

void FluxDistributionStepper::initialize()
{
  DifferentialStepper::initialize();
}


