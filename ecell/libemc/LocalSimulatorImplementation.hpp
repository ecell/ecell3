//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2007 Keio University
//       Copyright (C) 2005-2007 The Molecular Sciences Institute
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

#include "libecs/libecs.hpp"
#include "libecs/Model.hpp"

#include "libemc.hpp"
#include "SimulatorImplementation.hpp"


namespace libemc
{

  /** @defgroup libemc_module The Libemc Module 
   * This is the libemc module 
   * @{ 
   */ 
  
  class LocalSimulatorImplementation
    :
    public SimulatorImplementation
  {

  public:

    LocalSimulatorImplementation();
    LIBEMC_API virtual ~LocalSimulatorImplementation();

    LIBEMC_API virtual void createStepper( libecs::StringCref         aClassname,
				libecs::StringCref         anId );

    LIBEMC_API virtual void deleteStepper( libecs::StringCref anID );

    LIBEMC_API virtual const libecs::Polymorph getStepperList() const;

    LIBEMC_API virtual const libecs::Polymorph 
    getStepperPropertyList( libecs::StringCref aStepperID ) const;

    LIBEMC_API virtual const libecs::Polymorph 
    getStepperPropertyAttributes( libecs::StringCref aStepperID, 
				  libecs::StringCref aPropertyName ) const;

    LIBEMC_API virtual void setStepperProperty( libecs::StringCref    aStepperID,
				     libecs::StringCref    aPropertyName,
				     libecs::PolymorphCref aValue );

    LIBEMC_API virtual const libecs::Polymorph
    getStepperProperty( libecs::StringCref aStepperID,
			libecs::StringCref aPropertyName ) const;

    LIBEMC_API virtual void loadStepperProperty( libecs::StringCref    aStepperID,
				      libecs::StringCref    aPropertyName,
				      libecs::PolymorphCref aValue );

    LIBEMC_API virtual const libecs::Polymorph
    saveStepperProperty( libecs::StringCref aStepperID,
			 libecs::StringCref aPropertyName ) const;

    LIBEMC_API virtual const libecs::String
    getStepperClassName( libecs::StringCref aStepperID ) const;


    LIBEMC_API virtual const libecs::PolymorphMap
	   	 getClassInfo( libecs::StringCref aClasstype,
			       libecs::StringCref aClassname,
			       const libecs::Integer forceReload );

    
    LIBEMC_API virtual void createEntity( libecs::StringCref aClassname, 
			       libecs::StringCref aFullIDString );

    LIBEMC_API virtual void deleteEntity( libecs::StringCref aFullIDString );

    LIBEMC_API virtual const libecs::Polymorph 
    getEntityList( libecs::StringCref anEntityTypeString,
		   libecs::StringCref aSystemPathString ) const;

    LIBEMC_API virtual const libecs::Polymorph 
    getEntityPropertyList( libecs::StringCref aFullID ) const;

    LIBEMC_API virtual const bool isEntityExist( libecs::StringCref aFullIDString ) const;

    LIBEMC_API virtual void setEntityProperty( libecs::StringCref    aFullPNString,
				    libecs::PolymorphCref aValue );

    LIBEMC_API virtual const libecs::Polymorph
    getEntityProperty( libecs::StringCref aFullPNString ) const;

    LIBEMC_API virtual void loadEntityProperty( libecs::StringCref    aFullPNString,
				     libecs::PolymorphCref aValue );

    LIBEMC_API virtual const libecs::Polymorph
    saveEntityProperty( libecs::StringCref aFullPNString ) const;

    LIBEMC_API virtual const libecs::Polymorph
    getEntityPropertyAttributes( libecs::StringCref aFullPNString ) const;

    LIBEMC_API virtual const libecs::String
    getEntityClassName( libecs::StringCref aFullIDString ) const;

    LIBEMC_API virtual void createLogger( libecs::StringCref aFullPNString );

    LIBEMC_API virtual void createLogger( libecs::StringCref aFullPNString, libecs::Polymorph aParamList  );

    LIBEMC_API virtual const libecs::Polymorph getLoggerList() const;

    LIBEMC_API virtual const libecs::DataPointVectorSharedPtr 
    getLoggerData( libecs::StringCref aFullPNString ) const;

    LIBEMC_API virtual const libecs::DataPointVectorSharedPtr
    getLoggerData( libecs::StringCref aFullPNString, 
		   libecs::RealCref start, libecs::RealCref end ) const;

    LIBEMC_API virtual const libecs::DataPointVectorSharedPtr
    getLoggerData( libecs::StringCref aFullPNString,
		   libecs::RealCref start, libecs::RealCref end, 
		   libecs::RealCref interval ) const;

    LIBEMC_API virtual const libecs::Real 
    getLoggerStartTime( libecs::StringCref aFullPNString ) const;

    LIBEMC_API virtual const libecs::Real 
    getLoggerEndTime( libecs::StringCref aFullPNString ) const;

    LIBEMC_API virtual void setLoggerMinimumInterval( libecs::StringCref aFullPNString, 
					   libecs::RealCref anInterval );

    LIBEMC_API virtual const libecs::Real 
    getLoggerMinimumInterval( libecs::StringCref aFullPNString ) const;


    LIBEMC_API virtual void 
    setLoggerPolicy( libecs::StringCref aFullPNString, 
			      libecs::Polymorph aParamList ) ;

    LIBEMC_API virtual const libecs::Polymorph
    getLoggerPolicy( libecs::StringCref aFullPNString ) const;


    LIBEMC_API virtual const libecs::Logger::size_type 
    getLoggerSize( libecs::StringCref aFullPNString ) const;

    LIBEMC_API virtual const libecs::Polymorph getNextEvent() const;

    LIBEMC_API virtual void step( const libecs::Integer aNumSteps );

    LIBEMC_API virtual const libecs::Real getCurrentTime() const;

    LIBEMC_API virtual void run();

    LIBEMC_API virtual void run( const libecs::Real aDuration );

    LIBEMC_API virtual void stop();

    void clearEventChecker();

    LIBEMC_API virtual void setEventChecker( EventCheckerSharedPtrCref anEventChecker );

    LIBEMC_API virtual void setEventHandler( EventHandlerSharedPtrCref anEventHandler );

    LIBEMC_API virtual const libecs::Polymorph getDMInfo();

  protected:

    libecs::ModelRef getModel() 
    { 
      return theModel; 
    }

    libecs::ModelCref getModel() const 
    { 
      return theModel; 
    }

    void initialize() const;

    libecs::LoggerPtr getLogger( libecs::StringCref aFullPNString ) const;


    void setDirty()
    {
      theDirtyFlag = true;
    }

    const bool isDirty() const
    {
      return theDirtyFlag;
    }

    inline void handleEvent()
    {
      if( (*theEventChecker)() )
	{
	  do
	    {
	      (*theEventHandler)();
	    }	while( (*theEventChecker)() );
	  
	  clearDirty();
	}
    }

    void clearDirty() const
    {
      if( isDirty() )
        {
 	  initialize();
 	  theDirtyFlag = false;
 	}
    }

    void start()
    {
      clearDirty();
      theRunningFlag = true;
    }

    void initialize()
    {
      // theModel.initialize();
    }

    void runWithEvent();
    void runWithoutEvent();

  private:

    bool                       theRunningFlag;

    mutable bool               theDirtyFlag;

    libecs::Integer            theEventCheckInterval;

    libecs::Model              theModel;

    EventCheckerSharedPtr      theEventChecker;
    EventHandlerSharedPtr      theEventHandler;

  };  

  /** @} */ //end of libemc_module 

} // namespace libemc


#endif   /* __LOCALSIMULATORIMPLEMENTATION_HPP */
