//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 2000-2002 Keio University
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
// written by Masayuki Okayama <smash@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//


#include "libecs.hpp"

#include "Util.hpp"
#include "Logger.hpp"
#include "PropertyInterface.hpp"
#include "FullID.hpp"
#include "Model.hpp"
#include "PropertySlot.hpp"

#include "LoggerBroker.hpp"


namespace libecs
{

  LoggerBroker::LoggerBroker( ModelRef aModel )
    :
    theModel( aModel )
  {
    ; // do nothing
  }

  LoggerBroker::~LoggerBroker()
  {
    FOR_ALL_SECOND( LoggerMap, theLoggerMap, ~Logger );
  }


  void LoggerBroker::flush()
  {
    FOR_ALL_SECOND( LoggerMap, theLoggerMap, flush );
  }


  LoggerPtr 
  LoggerBroker::getLogger( FullPNCref aFullPN ) const
  {
    LoggerMapConstIterator aLoggerMapIterator( theLoggerMap.find( aFullPN ) );

    if( aLoggerMapIterator == theLoggerMap.end() )
      {
	THROW_EXCEPTION( NotFound, "Logger [" + aFullPN.getString() 
			 + "] not found." );
      }

    return aLoggerMapIterator->second;
  }

  LoggerPtr LoggerBroker::createLogger( FullPNCref aFullPN ) 
  {
    if( theLoggerMap.find( aFullPN ) != theLoggerMap.end() )
      {
	THROW_EXCEPTION( AlreadyExist, "Logger [" + aFullPN.getString()
			 + "] already exist." );
      }

    EntityPtr anEntityPtr( getModel().getEntity( aFullPN.getFullID() ) );

    const String aPropertyName( aFullPN.getPropertyName() );
    PropertySlotMapCref aPropertySlotMap( anEntityPtr->getPropertySlotMap() );

    PropertySlotMapConstIterator 
      aPropertySlotMapIterator( aPropertySlotMap.find( aPropertyName ) );

    if( aPropertySlotMapIterator == aPropertySlotMap.end() )
      {
	THROW_EXCEPTION( NotFound, "PropertySlot not found" );
      }

    PropertySlotPtr aPropertySlotPtr( aPropertySlotMapIterator->second );
    StepperPtr aStepperPtr( anEntityPtr->getStepper() );

    LoggerPtr aNewLogger( new Logger( *aPropertySlotPtr, *aStepperPtr ) );

    aPropertySlotPtr->connectLogger( aNewLogger );
    theLoggerMap[aFullPN] = aNewLogger;

    // don't forget this!
    aPropertySlotPtr->updateLogger();
    aNewLogger->flush();

    aStepperPtr->registerLoggedPropertySlot( aPropertySlotPtr );

    return aNewLogger;
  }

  

} // namespace libecs








