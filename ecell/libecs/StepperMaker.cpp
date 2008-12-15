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

#include "StepperMaker.hpp"

#include "DiscreteTimeStepper.hpp"
#include "DiscreteEventStepper.hpp"
#include "PassiveStepper.hpp"

namespace libecs
{
  StepperMaker::StepperMaker( PropertiedObjectMaker& maker )
    : theBackend( maker )
  {
    makeClassList();
  }

  StepperMaker::~StepperMaker()
  {
  }

  Stepper* StepperMaker::make( const std::string& aClassName )
  {
    const PropertiedObjectMaker::SharedModule& mod(
	theBackend.getModule( aClassName, false ) );
    if ( mod.getTypeName() != "Stepper" )
      {
	throw TypeError( "specified class is not a Stepper" );
      }
    return reinterpret_cast< Stepper* >( theBackend.make( aClassName ) );
  }

  const PropertiedObjectMaker::SharedModule& StepperMaker::getModule(
      const std::string& aClassName, bool forceReload )
  {
    const PropertiedObjectMaker::SharedModule& mod(
	theBackend.getModule( aClassName, forceReload ) );

    if ( mod.getTypeName() != "Stepper" )
      {
	throw TypeError( "specified class is not a Stepper" );
      }
    return mod;
  }

  void StepperMaker::makeClassList()
  {
    theBackend.NewDynamicModule( PropertiedClass, DiscreteEventStepper );
    theBackend.NewDynamicModule( PropertiedClass, DiscreteTimeStepper );
    theBackend.NewDynamicModule( PropertiedClass, PassiveStepper );
  }


} // namespace libecs
