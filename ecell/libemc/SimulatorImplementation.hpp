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


#ifndef __SIMULATORIMPLEMENTATION_HPP
#define __SIMULATORIMPLEMENTATION_HPP

#include "libecs/libecs.hpp"
#include "libecs/Message.hpp"

#include "libemc.hpp"

namespace libemc
{

  /** @defgroup libemc_module The Libemc Module 
   * This is the libemc module 
   * @{ 
   */ 
  
  /**
     Pure virtual base class (interface definition) of simulator
     implementation.
  */

  class SimulatorImplementation
  {

  public:

    SimulatorImplementation() {}
    virtual ~SimulatorImplementation() {}

    virtual libecs::RootSystemRef getRootSystem() = 0;

    virtual void createEntity( libecs::StringCref    classname, 
			       libecs::PrimitiveType type,
			       libecs::StringCref    systempath,
			       libecs::StringCref    id,
			       libecs::StringCref    name ) = 0;

    virtual void setProperty( libecs::PrimitiveType       type,
			      libecs::StringCref          systempath,
			      libecs::StringCref          id,
			      libecs::StringCref          propertyname,
			      libecs::UVariableVectorCref data ) = 0;

    virtual const libecs::UVariableVectorRCPtr
    getProperty( libecs::PrimitiveType type,
		 libecs::StringCref    systempath,
		 libecs::StringCref    id,
		 libecs::StringCref    propertyname ) = 0;

    virtual void step() = 0;

    virtual void initialize() = 0;

    virtual libecs::LoggerPtr 
    getLogger( libecs::PrimitiveType type,
	       libecs::StringCref    systempath,
	       libecs::StringCref    id,
	       libecs::StringCref    propertyname ) = 0;

    virtual libecs::StringVectorRCPtr getLoggerList() = 0;

    virtual void run() = 0;

    virtual void run( libecs::Real aDuration ) = 0;

    virtual void stop() = 0;

    virtual void setPendingEventChecker( PendingEventCheckerFuncPtr aPendingEventChecker ) = 0;

    virtual void setEventHandler( EventHandlerFuncPtr anEventHandler ) = 0;

  };   //end of class Simulator

  /** @} */ //end of libemc_module 

} // namespace libemc

#endif   /* ___SIMULATOR_IMPLEMENTATION_H___ */

