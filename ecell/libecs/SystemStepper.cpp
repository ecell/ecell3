//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2008 Keio University
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

#include <limits>
#include <boost/foreach.hpp>

#include "Model.hpp"
#include "System.hpp"
#include "Variable.hpp"

#include "SystemStepper.hpp"

#include "VariableValueIntegrator.hpp"

namespace libecs
{

LIBECS_DM_INIT_STATIC( SystemStepper, Stepper );

////////////////////////// Stepper

void SystemStepper::startup()
{
    setCurrentTime( INF );
    setMaxStepInterval( INF );
    setStepInterval( INF );
    setPriority( ( std::numeric_limits<Integer>::max )() );
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

void SystemStepper::integrateVariablesRecursively( System* const aSystem,
        RealParam aTime )
{
    BOOST_FOREACH( const System::VariablesCRange::value_type& i, aSystem->getBelongings<Variable>() )
    {
        Variable* const aVariable( i.second );

        if ( aVariable->isIntegrationNeeded() )
        {
            aVariable->getVariableValueIntegrator()->integrate( aTime );
        }
    }

    BOOST_FOREACH( const System::SystemsCRange::value_type& i, aSystem->getBelongings<System>() )
    {
        System* const aSubSystem( i.second );
        integrateVariablesRecursively( aSubSystem, aTime );
    }
}

void SystemStepper::initialize()
{
    ; // do nothing
}

} // namespace libecs

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
