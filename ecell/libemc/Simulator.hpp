//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
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


#ifndef __SIMULATOR_HPP
#define __SIMULATOR_HPP

#include "libecs/libecs.hpp"
#include "libecs/Message.hpp"
#include "libecs/EntityType.hpp"

#include "libemc.hpp"
#include "EmcLogger.hpp"
#include "SimulatorImplementation.hpp"

namespace libemc
{
  
  /** @defgroup libemc_module The Libemc Module 
   * This is the libemc module 
   * @{ 
   */ 
  

  class Simulator
  {

  public:

    Simulator();
    virtual ~Simulator() {}

    void createStepper( libecs::StringCref          aClassname,
			libecs::StringCref          anId,
			libecs::UVariableVectorCref aData )
    {
      theSimulatorImplementation->createStepper( aClassname, anId, aData );
    }


    void createEntity( libecs::StringCref           aClassname, 
		       libecs::StringCref           aFullIDString,
		       libecs::StringCref           aName )
    {
      theSimulatorImplementation->createEntity( aClassname,
						aFullIDString,
						aName );
    }

    void setProperty( libecs::StringCref            aFullPNString,
		      libecs::UVariableVectorCref   aData )
    {
      theSimulatorImplementation->setProperty( aFullPNString,
					       aData );
    }

    const libecs::UVariableVectorRCPtr
    getProperty( libecs::StringCref aFullPNString )
    {
      return theSimulatorImplementation->getProperty( aFullPNString );
    }

    EmcLogger getLogger( libecs::StringCref aFullPNString )
    {
      return theSimulatorImplementation->getLogger( aFullPNString );
    }

    void step()
    {
      theSimulatorImplementation->step();
    }

    void initialize()
    {
      theSimulatorImplementation->initialize();
    }

    const libecs::Real getCurrentTime()
    {
      return theSimulatorImplementation->getCurrentTime();
    }

    libecs::StringVectorRCPtr getLoggerList()
    {
      return theSimulatorImplementation->getLoggerList();
    }

    void run()
    {
      theSimulatorImplementation->run();
    }

    void run( libecs::Real aDuration )
    {
      theSimulatorImplementation->run( aDuration );
    }

    void stop()
    {
      theSimulatorImplementation->stop();
    }

    void setPendingEventChecker( PendingEventCheckerPtr aPendingEventChecker )
    {
       theSimulatorImplementation->
	 setPendingEventChecker( aPendingEventChecker );
     }

    void setEventHandler( EventHandlerPtr anEventHandler )
    {
       theSimulatorImplementation->setEventHandler( anEventHandler );
    }

  private:

    SimulatorImplementation* theSimulatorImplementation;

  };

  /** @} */ //end of libemc_module 

} // namespace libemc

#endif   /* __SIMULATOR_HPP */





