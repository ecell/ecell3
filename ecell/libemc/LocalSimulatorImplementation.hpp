//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2000 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-CELL is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-CELL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-CELL -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Kouichi Takahashi <shafi@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//


#ifndef __LOCALSIMULATORIMPLEMENTATION_HPP
#define __LOCALSIMULATORIMPLEMENTATION_HPP

#include "libecs/libecs.hpp"

#include "libemc.hpp"
#include "SimulatorImplementation.hpp"

namespace libemc
{

  class LocalSimulatorImplementation
    :
    public SimulatorImplementation
  {

  public:

    LocalSimulatorImplementation();
    virtual ~LocalSimulatorImplementation();

    libecs::RootSystemRef   getRootSystem() 
    { 
      return theRootSystem; 
    }

    libecs::LoggerBrokerRef getLoggerBroker()
    { 
      return theLoggerBroker; 
    }

    virtual void createEntity( libecs::StringCref    classname, 
			       libecs::PrimitiveType type,
			       libecs::StringCref    systempath,
			       libecs::StringCref    id,
			       libecs::StringCref    name );

    virtual void setProperty( libecs::PrimitiveType       type,
			      libecs::StringCref          systempath,
			      libecs::StringCref          id,
			      libecs::StringCref          propertyname,
			      libecs::UConstantVectorCref data );

    virtual const libecs::UConstantVectorRCPtr
    getProperty( libecs::PrimitiveType type,
		 libecs::StringCref    systempath,
		 libecs::StringCref    id,
		 libecs::StringCref    propertyname );

    void step();

    void initialize();

    virtual libecs::LoggerPtr getLogger( libecs::PrimitiveType type,
					 libecs::StringCref    systempath,
					 libecs::StringCref    id,
					 libecs::StringCref    propertyname );

    virtual libecs::StringVector getLoggerList();

    virtual void run();

    virtual void stop();

    virtual void setPendingEventChecker( PendingEventCheckerFuncPtr
					 aPendingEventChecker );

    void clearPendingEventChecker();

    virtual void setEventHandler( EventHandlerFuncPtr anEventHandler );

  private:

    static bool defaultPendingEventChecker();

  private:

    libecs::RootSystemRef      theRootSystem;
    libecs::LoggerBrokerRef    theLoggerBroker;
    bool                       theRunningFlag;
    PendingEventCheckerFuncPtr thePendingEventChecker;
    EventHandlerFuncPtr        theEventHandler;


  };  


} // namespace libemc


#endif   /* __LOCALSIMULATORIMPLEMENTATION_HPP */
