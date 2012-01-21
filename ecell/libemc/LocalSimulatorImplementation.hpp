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

#ifndef __LOCALSIMULATORIMPLEMENTATION_HPP
#define __LOCALSIMULATORIMPLEMENTATION_HPP

#ifdef DLL_EXPORT
#undef DLL_EXPORT
#define _DLL_EXPORT
#endif /* DLL_EXPORT */

#include "libecs/libecs.hpp"
#include "libecs/Model.hpp"

#ifdef _DLL_EXPORT
#define DLL_EXPORT
#undef _DLL_EXPORT
#endif /* _DLL_EXPORT */

#include "libemc.hpp"
#include "SimulatorImplementation.hpp"

namespace libemc
{
class LIBEMC_API LocalSimulatorImplementation
    : public SimulatorImplementation
{
public:
    LocalSimulatorImplementation();
    virtual ~LocalSimulatorImplementation();

    virtual void createStepper( libecs::String const& aClassname,
                                libecs::String const& anId );

    virtual void deleteStepper( libecs::String const& anID );

    virtual libecs::Polymorph getStepperList() const;

    virtual libecs::Polymorph 
    getStepperPropertyList( libecs::String const& aStepperID ) const;

    virtual libecs::Polymorph 
    getStepperPropertyAttributes( libecs::String const& aStepperID, 
                                  libecs::String const& aPropertyName ) const;

    virtual void setStepperProperty( libecs::String const& aStepperID,
                                     libecs::String const& aPropertyName,
                                     libecs::Polymorph const& aValue );

    virtual libecs::Polymorph
    getStepperProperty( libecs::String const& aStepperID,
                        libecs::String const& aPropertyName ) const;

    virtual void loadStepperProperty( libecs::String const& aStepperID,
                                      libecs::String const& aPropertyName,
                                      libecs::Polymorph const& aValue );

    virtual libecs::Polymorph
    saveStepperProperty( libecs::String const& aStepperID,
                         libecs::String const& aPropertyName ) const;

    virtual libecs::String
    getStepperClassName( libecs::String const& aStepperID ) const;


    virtual SimulatorImplementation::PolymorphMap getClassInfo(
            libecs::String const& aClassname ) const;

    
    virtual void createEntity( libecs::String const& aClassname, 
                               libecs::String const& aFullIDString );

    virtual void deleteEntity( libecs::String const& aFullIDString );

    virtual libecs::Polymorph 
    getEntityList( libecs::String const& anEntityTypeString,
                   libecs::String const& aSystemPathString ) const;

    virtual libecs::Polymorph 
    getEntityPropertyList( libecs::String const& aFullID ) const;

    virtual bool entityExists( libecs::String const& aFullIDString ) const;

    virtual void setEntityProperty( libecs::String const& aFullPNString,
                                    libecs::Polymorph const& aValue );

    virtual libecs::Polymorph
    getEntityProperty( libecs::String const& aFullPNString ) const;

    virtual void loadEntityProperty( libecs::String const& aFullPNString,
                                     libecs::Polymorph const& aValue );

    virtual libecs::Polymorph
    saveEntityProperty( libecs::String const& aFullPNString ) const;

    virtual libecs::Polymorph
    getEntityPropertyAttributes( libecs::String const& aFullPNString ) const;

    virtual libecs::String
    getEntityClassName( libecs::String const& aFullIDString ) const;

    virtual void createLogger( libecs::String const& aFullPNString );

    virtual void createLogger( libecs::String const& aFullPNString,
                               libecs::Polymorph aParamList );

    virtual libecs::Polymorph getLoggerList() const;

    virtual boost::shared_ptr< libecs::DataPointVector > 
    getLoggerData( libecs::String const& aFullPNString ) const;

    virtual boost::shared_ptr< libecs::DataPointVector >
    getLoggerData( libecs::String const& aFullPNString, 
                   libecs::Real start, libecs::Real end ) const;

    virtual boost::shared_ptr< libecs::DataPointVector >
    getLoggerData( libecs::String const& aFullPNString,
                   libecs::Real start, libecs::Real end, 
                   libecs::Real interval ) const;

    virtual libecs::Real 
    getLoggerStartTime( libecs::String const& aFullPNString ) const;

    virtual libecs::Real 
    getLoggerEndTime( libecs::String const& aFullPNString ) const;

    virtual void 
    setLoggerPolicy( libecs::String const& aFullPNString, 
                     libecs::Polymorph aParamList ) ;

    virtual libecs::Polymorph
    getLoggerPolicy( libecs::String const& aFullPNString ) const;


    virtual libecs::Logger::size_type 
    getLoggerSize( libecs::String const& aFullPNString ) const;

    virtual libecs::Polymorph getNextEvent() const;

    virtual void step( libecs::Integer aNumSteps );

    virtual libecs::Real getCurrentTime() const;

    virtual void run();

    virtual void run( libecs::Real aDuration );

    virtual void stop();

    void clearEventChecker();

    virtual void setEventChecker( boost::shared_ptr< EventChecker > const& anEventChecker );

    virtual void setEventHandler( boost::shared_ptr< EventHandler > const& anEventHandler );

    virtual libecs::PolymorphVector getDMInfo() const;

    virtual SimulatorImplementation::PolymorphMap
    getPropertyInfo( libecs::String const& aClassname ) const;

    virtual char getDMSearchPathSeparator() const;

    virtual libecs::String getDMSearchPath() const;

    virtual void setDMSearchPath( libecs::String const& aDMSearchPath );

protected:

    libecs::Model& getModel() 
    { 
        return theModel; 
    }

    libecs::Model const& getModel() const 
    { 
        return theModel; 
    }

    void initialize() const;

    libecs::Logger* getLogger( libecs::String const& aFullPNString ) const;


    void setDirty()
    {
        theModel.markDirty();
    }

    const bool isDirty() const
    {
        return theModel.isDirty();
    }

    inline void handleEvent()
    {
        while ( (*theEventChecker)() )
        {
            (*theEventHandler)();
        }
    }

    void start()
    {
        theRunningFlag = true;
    }

    void runWithEvent( libecs::Real aStopTime );
    void runWithoutEvent( libecs::Real aStopTime );

    static libecs::Polymorph buildPolymorph( libecs::Logger::Policy const& );
    static libecs::Polymorph buildPolymorph( libecs::PropertyAttributes const& );

private:

    bool                    theRunningFlag;

    mutable bool            theDirtyFlag;

    libecs::Integer         theEventCheckInterval;

    ModuleMaker< libecs::EcsObject >* thePropertiedObjectMaker;
    libecs::Model           theModel;

    boost::shared_ptr< EventChecker > theEventChecker;
    boost::shared_ptr< EventHandler > theEventHandler;

};

} // namespace libemc


#endif   /* __LOCALSIMULATORIMPLEMENTATION_HPP */
