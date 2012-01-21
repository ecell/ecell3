//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2012 Keio University
//       Copyright (C) 2005-2009 The Molecular Sciences Institute
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

#ifndef __SIMULATORIMPLEMENTATION_HPP
#define __SIMULATORIMPLEMENTATION_HPP

#ifdef DLL_EXPORT
#undef DLL_EXPORT
#define _DLL_EXPORT
#endif /* DLL_EXPORT */

#include "libecs/libecs.hpp"
#include "libecs/Logger.hpp"

#ifdef _DLL_EXPORT
#define DLL_EXPORT
#undef _DLL_EXPORT
#endif /* _DLL_EXPORT */

#include <map>
#include "libemc.hpp"

namespace libemc
{
/**
   Pure virtual base class (interface definition) of simulator implementation.
*/

class LIBEMC_API SimulatorImplementation
{
public:
    typedef std::map< libecs::String, libecs::Polymorph > PolymorphMap;

public:

    SimulatorImplementation() {}
    virtual ~SimulatorImplementation() {}

    virtual void createStepper( libecs::String const&  aClassname,
                                 libecs::String const& anId ) = 0;

    virtual void deleteStepper( libecs::String const& anID ) = 0;

    virtual libecs::Polymorph getStepperList() const = 0;

    virtual libecs::Polymorph 
    getStepperPropertyList( libecs::String const& aStepperID ) const = 0;

    virtual libecs::Polymorph 
    getStepperPropertyAttributes( libecs::String const& aStepperID, 
                                  libecs::String const& aPropertyName ) const = 0;

    virtual void setStepperProperty( libecs::String const& aStepperID,
                                     libecs::String const& aPropertyName,
                                     libecs::Polymorph const& aValue ) = 0;

    virtual libecs::Polymorph
    getStepperProperty( libecs::String const& aStepperID,
                        libecs::String const& aPropertyName ) const = 0;

    virtual void loadStepperProperty( libecs::String const&    aStepperID,
                                      libecs::String const&    aPropertyName,
                                      libecs::Polymorph const& aValue ) = 0;

    virtual libecs::Polymorph
    saveStepperProperty( libecs::String const& aStepperID,
                         libecs::String const& aPropertyName ) const = 0;

    virtual libecs::String
    getStepperClassName( libecs::String const& aStepperID ) const = 0;

    
    virtual void createEntity( libecs::String const& aClassname, 
                               libecs::String const& aFullIDString ) = 0;

    virtual void deleteEntity( libecs::String const& aFullIDString ) = 0;

    virtual libecs::Polymorph 
    getEntityList( libecs::String const& anEntityTypeString,
                   libecs::String const& aSystemPathString ) const = 0;

    virtual libecs::Polymorph 
    getEntityPropertyList( libecs::String const& aFullIDString ) const = 0;

    virtual bool 
    entityExists( libecs::String const& aFullIDString ) const = 0;

    virtual void setEntityProperty( libecs::String const& aFullPNString,
                                    libecs::Polymorph const& aValue ) = 0;

    virtual libecs::Polymorph
    getEntityProperty( libecs::String const& aFullPNString ) const = 0;

    virtual void loadEntityProperty( libecs::String const& aFullPNString,
                                     libecs::Polymorph const& aValue ) = 0;

    virtual libecs::Polymorph
    saveEntityProperty( libecs::String const& aFullPNString ) const = 0;

    virtual libecs::Polymorph
    getEntityPropertyAttributes( libecs::String const& aFullPNString ) const = 0;

    virtual libecs::String
    getEntityClassName( libecs::String const& aFullIDString ) const = 0;

    virtual void createLogger( libecs::String const& aFullPNString ) = 0;

    virtual void createLogger( libecs::String const& aFullPNString,
                               libecs::Polymorph aParamList    ) = 0;

    virtual libecs::Polymorph getLoggerList() const = 0;

    virtual boost::shared_ptr< libecs::DataPointVector > 
    getLoggerData( libecs::String const& aFullPNString ) const = 0;

    virtual boost::shared_ptr< libecs::DataPointVector >
    getLoggerData( libecs::String const& aFullPNString, 
                   libecs::Real aStartTime, 
                   libecs::Real anEndTime ) const = 0;

    virtual boost::shared_ptr< libecs::DataPointVector >
    getLoggerData( libecs::String const& aFullPNString,
                   libecs::Real aStartTime, libecs::Real anEndTime, 
                   libecs::Real interval ) const = 0;

    virtual libecs::Real 
    getLoggerStartTime( libecs::String const& aFullPNString ) const = 0;

    virtual libecs::Real 
    getLoggerEndTime( libecs::String const& aFullPNString ) const = 0;

    virtual void 
    setLoggerPolicy( libecs::String const& aFullPNString, 
                     libecs::Polymorph aParamList ) = 0;

    virtual libecs::Polymorph
    getLoggerPolicy( libecs::String const& aFullPNString ) const = 0;

    virtual libecs::Logger::size_type
    getLoggerSize( libecs::String const& aFullPNString ) const = 0;

    virtual libecs::Polymorph getNextEvent() const = 0;

    virtual void step( libecs::Integer aNumSteps ) = 0;

    virtual libecs::Real getCurrentTime() const = 0;

    virtual void run() = 0;

    virtual void run( libecs::Real aDuration ) = 0;

    virtual void stop() = 0;

    virtual void setEventChecker( boost::shared_ptr< EventChecker > const& aEventChecker ) = 0;

    virtual void setEventHandler( boost::shared_ptr< EventHandler > const& anEventHandler ) = 0;

    virtual PolymorphMap
    getClassInfo( libecs::String const& aClassname ) const = 0;

    virtual PolymorphMap
    getPropertyInfo( libecs::String const& aClassname ) const = 0; 

    virtual libecs::PolymorphVector getDMInfo() const = 0;

    virtual char getDMSearchPathSeparator() const = 0;

    virtual libecs::String getDMSearchPath() const = 0;

    virtual void setDMSearchPath( libecs::String const& aDMSearchPath ) = 0;

}; //end of class Simulator

} // namespace libemc

#endif   /* ___SIMULATOR_IMPLEMENTATION_H___ */
