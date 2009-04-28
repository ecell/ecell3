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
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include "AdaptiveDifferentialStepper.hpp"

namespace libecs
{
LIBECS_DM_INIT_STATIC( AdaptiveDifferentialStepper, Stepper );

AdaptiveDifferentialStepper::AdaptiveDifferentialStepper()
    : theTolerance( 1.0e-6 ),
      theAbsoluteToleranceFactor( 1.0 ),
      theStateToleranceFactor( 1.0 ),
      theDerivativeToleranceFactor( 1.0 ),
      theEpsilonChecked( 0 ),
      theAbsoluteEpsilon( 0.1 ),
      theRelativeEpsilon( 0.1 ),
      safety( 0.9 ),
      theMaxErrorRatio( 1.0 )
{
    // use more narrow range
    setMinStepInterval( 1e-100 );
    setMaxStepInterval( 1e+10 );
}

AdaptiveDifferentialStepper::~AdaptiveDifferentialStepper()
{
    ; // do nothing
}


void AdaptiveDifferentialStepper::initialize()
{
    DifferentialStepper::initialize();
}

void AdaptiveDifferentialStepper::step()
{
    theStateFlag = false;

    clearVariables();

    setStepInterval( getNextStepInterval() );

    while ( !calculate() )
    {
        const Real anExpectedStepInterval( safety * getStepInterval() 
                                           * pow( getMaxErrorRatio(),
                                                  -1.0 / getOrder() ) );

        if ( anExpectedStepInterval > getMinStepInterval() )
        {
            // shrink it if the error exceeds 110%
            setStepInterval( anExpectedStepInterval );
        }
        else
        {
            setStepInterval( getMinStepInterval() );

            // this must return false,
            // so theTolerableStepInterval does NOT LIMIT the error.
            THROW_EXCEPTION_INSIDE( SimulationError,
                                    asString() + ": the error-limit step "
                                    " interval is too small" );

            calculate();
            break;
        }
    }

    // an extra calculation for resetting the activities of processes
    fireProcesses();

    setTolerableStepInterval( getStepInterval() );

    theStateFlag = true;

    // grow it if error is 50% less than desired
    const Real maxError( getMaxErrorRatio() );
    if ( maxError < 0.5 )
    {
        const Real aNewStepInterval( getTolerableStepInterval() * safety
                                     * pow( maxError,
                                            -1.0 / ( getOrder() + 1 ) ) );
        setNextStepInterval( aNewStepInterval );
    }
    else 
    {
        setNextStepInterval( getTolerableStepInterval() );
    }
}

} // namespace libecs
