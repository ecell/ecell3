//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 2000-2001 Keio University
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
#include "RootSystem.hpp"

#include "LoggerBroker.hpp"


namespace libecs
{

  LoggerBroker::LoggerBroker( RootSystemRef aRootSystem )
    :
    theRootSystem( aRootSystem ),
    theGetCurrentTimeMethod( theRootSystem, &RootSystem::getCurrentTime )
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
    EntityPtr anEntityPtr(theRootSystem.getEntity( fpn.getFullID() ));

    String aPropertyName( fpn.getPropertyName() );

    PropertyMapConstIterator 
      aPropertyMapIterator( anEntityPtr->getPropertySlotMap().
			    find( aPropertyName ) );

    if( aPropertyMapIterator == anEntityPtr->getPropertySlotMap().end() )
      {
	throw NotFound( "not found" );
      }

    //    aPropertyMapIterator->second->getProxy()->setLogger( aLoggerPtr );

    //    appendLogger( aLoggerPtr );
    theLoggerMap[fpn] = new Logger( theGetCurrentTimeMethod,
				    *aPropertyMapIterator->second );
    aPropertyMapIterator->second->setLogger(theLoggerMap[fpn]);

    anEntityPtr->getSuperSystem()
      ->getStepper()->registerPropertySlot( aPropertyMapIterator->second );

    return theLoggerMap[fpn];
  }

  /*
  void LoggerBroker::appendLogger( LoggerPtr logger )
  {
    theLoggerMap[logger->getName()] = logger;
  }
  */
  

} // namespace libecs








