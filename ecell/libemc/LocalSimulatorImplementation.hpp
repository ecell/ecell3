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


#ifndef __LOCALSIMULATORIMPLEMENTATION_HPP
#define __LOCALSIMULATORIMPLEMENTATION_HPP

#include "libecs/libecs.hpp"

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
    virtual ~LocalSimulatorImplementation();

    virtual void createStepper( libecs::StringCref          aClassname,
				libecs::StringCref          anId );


    virtual void setStepperProperty( libecs::StringCref          aStepperID,
				     libecs::StringCref          aPropertyName,
				     libecs::UVariableVectorCref aValue );

    virtual const libecs::UVariableVectorRCPtr
    getStepperProperty( libecs::StringCref aStepperID,
			libecs::StringCref aPropertyName );


    virtual void createEntity( libecs::StringCref           aClassname, 
			       libecs::StringCref           aFullIDString,
			       libecs::StringCref           aName );

    virtual void setProperty( libecs::StringCref            aFullPNString,
			      libecs::UVariableVectorCref   aData );

    virtual const libecs::UVariableVectorRCPtr
    getProperty( libecs::StringCref aFullPNString );

    virtual EmcLogger getLogger( libecs::StringCref aFullPNString );

    void step();

    void initialize();

    virtual const libecs::Real getCurrentTime();

    virtual libecs::StringVectorRCPtr getLoggerList();

    virtual void run();

    virtual void run( libecs::Real aDuration );

    virtual void stop();

    virtual void setPendingEventChecker( PendingEventCheckerPtr
					 aPendingEventChecker );

    void clearPendingEventChecker();

    virtual void setEventHandler( EventHandlerPtr anEventHandler );


  protected:

    libecs::ModelRef getModel() 
    { 
      return theModel; 
    }

  private:

    void runWithEvent( libecs::Real aDuration );
    void runWithoutEvent( libecs::Real aDuration );

  private:

    libecs::ModelRef           theModel;

    bool                       theRunningFlag;
    PendingEventCheckerPtr     thePendingEventChecker;
    EventHandlerPtr            theEventHandler;

  };  

  /** @} */ //end of libemc_module 

} // namespace libemc


#endif   /* __LOCALSIMULATORIMPLEMENTATION_HPP */
