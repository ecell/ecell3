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

#include "Logger.hpp"
#include "PropertyInterface.hpp"
#include "FullID.hpp"
#include "Model.hpp"

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
    for( LoggerMapIterator i( theLoggerMap.begin() );
	 i != theLoggerMap.end() ; ++i )
      {
	delete i->second;
      }
  }


  LoggerPtr LoggerBroker::getLogger( FullPNCref fpn )
  {
    LoggerMapIterator aLoggerMapIterator( theLoggerMap.find( fpn ) );
    if( aLoggerMapIterator != theLoggerMap.end() )
      {
	return aLoggerMapIterator->second;
      }
    else
      {
	return createLogger( fpn );
      }
  }

  LoggerPtr LoggerBroker::createLogger( FullPNCref fpn )
  {
    EntityPtr anEntityPtr( getModel().getEntity( fpn.getFullID() ) );

    String aPropertyName( fpn.getPropertyName() );

    PropertySlotMapConstIterator 
      aPropertySlotMapIterator( anEntityPtr->getPropertySlotMap().
				find( aPropertyName ) );

    if( aPropertySlotMapIterator == anEntityPtr->getPropertySlotMap().end() )
      {
	THROW_EXCEPTION( NotFound, "PropertySlot not found" );
      }

    PropertySlotPtr aPropertySlotPtr( aPropertySlotMapIterator->second );

    //    LoggerPtr aNewLogger( new Logger( theModel, *(aPropertySlotPtr) ) );
    LoggerPtr aNewLogger( new Logger( *( anEntityPtr->getStepper() ),
				      *(aPropertySlotPtr) ) );
    aPropertySlotPtr->connectLogger( aNewLogger );
    theLoggerMap[fpn] = aNewLogger;

    // don't forget this!
    aPropertySlotPtr->updateLogger();
    aNewLogger->flush();

    anEntityPtr->getSuperSystem()
      ->getStepper()->registerPropertySlot( aPropertySlotMapIterator->second );

    return aNewLogger;
  }

  /*
  void LoggerBroker::appendLogger( LoggerPtr logger )
  {
    theLoggerMap[logger->getName()] = logger;
  }
  */
  

} // namespace libecs








