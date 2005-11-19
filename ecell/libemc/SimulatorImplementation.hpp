//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
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


#ifndef __SIMULATORIMPLEMENTATION_HPP
#define __SIMULATORIMPLEMENTATION_HPP

#include "libecs/libecs.hpp"

#include "libemc.hpp"


namespace libemc
{

  /** @addtogroup libemc_module 
   * @{ 
   */ 

  /** @file */


  /**
     Pure virtual base class (interface definition) of simulator
     implementation.
  */

  class SimulatorImplementation
  {

  public:

    SimulatorImplementation() {}
    virtual ~SimulatorImplementation() {}


    virtual void createStepper( libecs::StringCref         aClassname,
				libecs::StringCref         anId ) = 0;

    virtual void deleteStepper( libecs::StringCref anID ) = 0;

    virtual const libecs::Polymorph getStepperList() const = 0;

    virtual const libecs::Polymorph 
    getStepperPropertyList( libecs::StringCref aStepperID ) const = 0;

    virtual const libecs::Polymorph 
    getStepperPropertyAttributes( libecs::StringCref aStepperID, 
				  libecs::StringCref aPropertyName ) const = 0;

    virtual void setStepperProperty( libecs::StringCref    aStepperID,
				     libecs::StringCref    aPropertyName,
				     libecs::PolymorphCref aValue ) = 0;

    virtual const libecs::Polymorph
    getStepperProperty( libecs::StringCref aStepperID,
			libecs::StringCref aPropertyName ) const = 0;

    virtual void loadStepperProperty( libecs::StringCref    aStepperID,
				      libecs::StringCref    aPropertyName,
				      libecs::PolymorphCref aValue ) = 0;

    virtual const libecs::Polymorph
    saveStepperProperty( libecs::StringCref aStepperID,
			 libecs::StringCref aPropertyName ) const = 0;

    virtual const libecs::String
    getStepperClassName( libecs::StringCref aStepperID ) const = 0;


    virtual const libecs::PolymorphMap
	    	  getClassInfo( libecs::StringCref aClasstype,
		    		libecs::StringCref aClassname,
		    		const libecs::Integer forceReload  ) = 0;

    
    virtual void createEntity( libecs::StringCref   aClassname, 
			       libecs::StringCref   aFullIDString ) = 0;

    virtual void deleteEntity( libecs::StringCref aFullIDString ) = 0;

    virtual const libecs::Polymorph 
    getEntityList( libecs::StringCref anEntityTypeString,
		   libecs::StringCref aSystemPathString ) const = 0;

    virtual const libecs::Polymorph 
    getEntityPropertyList( libecs::StringCref aFullIDString ) const = 0;

    virtual const bool 
    isEntityExist( libecs::StringCref  aFullIDString ) const = 0;

    virtual void setEntityProperty( libecs::StringCref    aFullPNString,
				    libecs::PolymorphCref aValue ) = 0;

    virtual const libecs::Polymorph
    getEntityProperty( libecs::StringCref aFullPNString ) const = 0;

    virtual void loadEntityProperty( libecs::StringCref    aFullPNString,
				     libecs::PolymorphCref aValue ) = 0;

    virtual const libecs::Polymorph
    saveEntityProperty( libecs::StringCref aFullPNString ) const = 0;

    virtual const libecs::Polymorph
    getEntityPropertyAttributes( libecs::StringCref aFullPNString ) const = 0;

    virtual const libecs::String
    getEntityClassName( libecs::StringCref aFullIDString ) const = 0;

    virtual void createLogger( libecs::StringCref aFullPNString ) = 0;

    virtual void createLogger( libecs::StringCref aFullPNString, libecs::Polymorph aParamList  ) = 0;

    virtual const libecs::Polymorph getLoggerList() const = 0;

    virtual const libecs::DataPointVectorSharedPtr 
    getLoggerData( libecs::StringCref aFullPNString ) const = 0;

    virtual const libecs::DataPointVectorSharedPtr
    getLoggerData( libecs::StringCref aFullPNString, 
		   libecs::RealCref aStartTime, 
		   libecs::RealCref anEndTime ) const = 0;

    virtual const libecs::DataPointVectorSharedPtr
    getLoggerData( libecs::StringCref aFullPNString,
		   libecs::RealCref aStartTime, libecs::RealCref anEndTime, 
		   libecs::RealCref interval ) const = 0;

    virtual const libecs::Real 
    getLoggerStartTime( libecs::StringCref aFullPNString ) const = 0;

    virtual const libecs::Real 
    getLoggerEndTime( libecs::StringCref aFullPNString ) const = 0;

    virtual void 
    setLoggerMinimumInterval( libecs::StringCref aFullPNString, 
			      libecs::RealCref anInterval ) = 0;

    virtual const libecs::Real 
    getLoggerMinimumInterval( libecs::StringCref aFullPNString ) const = 0;

    virtual void 
    setLoggerPolicy( libecs::StringCref aFullPNString, 
			      libecs::Polymorph aParamList ) = 0;

    virtual const libecs::Polymorph
    getLoggerPolicy( libecs::StringCref aFullPNString ) const = 0;

    virtual const libecs::Integer 
    getLoggerSize( libecs::StringCref aFullPNString ) const = 0;

    virtual const libecs::Polymorph getNextEvent() const = 0;

    virtual void step( const libecs::Integer aNumSteps ) = 0;

    virtual const libecs::Real getCurrentTime() const = 0;

    virtual void run() = 0;

    virtual void run( const libecs::Real aDuration ) = 0;

    virtual void stop() = 0;

    virtual void setEventChecker( EventCheckerSharedPtrCref aEventChecker ) = 0;

    virtual void setEventHandler( EventHandlerSharedPtrCref anEventHandler ) = 0;

    virtual const libecs::Polymorph getDMInfo() = 0;

  };   //end of class Simulator

  /** @} */ //end of libemc_module 

} // namespace libemc

#endif   /* ___SIMULATOR_IMPLEMENTATION_H___ */

