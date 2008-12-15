//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2008 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
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
// written by Masayuki Okayama <smash@e-cell.org>,
// E-Cell Project.
//
#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include "libecs.hpp"

#include "Entity.hpp"
#include "Util.hpp"
#include "Logger.hpp"
#include "PropertyInterface.hpp"
#include "FullID.hpp"
#include "Model.hpp"
#include "PropertySlot.hpp"
#include "PropertySlotProxyLoggerAdapter.hpp"
#include "LoggerBroker.hpp"


namespace libecs
{

  LoggerBroker::LoggerBroker()
    : theModel(0)
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


  LoggerPtr LoggerBroker::createLogger( FullPNCref aFullPN,   PolymorphVectorCref aParamList ) 
  {
    if( theLoggerMap.find( aFullPN ) != theLoggerMap.end() )
      {
	THROW_EXCEPTION( AlreadyExist, "Logger [" + aFullPN.getString()
			 + "] already exist." );
      }

    EntityPtr anEntityPtr( theModel->getEntity( aFullPN.getFullID() ) );

    const String aPropertyName( aFullPN.getPropertyName() );

    PropertySlotProxyPtr 
      aPropertySlotProxy( anEntityPtr->
			  createPropertySlotProxy( aPropertyName ) );

    LoggerAdapterPtr aLoggerAdapter
      ( new PropertySlotProxyLoggerAdapter( aPropertySlotProxy ) );


    LoggerPtr aNewLogger( new Logger( aLoggerAdapter) );

    anEntityPtr->registerLogger( aNewLogger );
    theLoggerMap[aFullPN] = aNewLogger;
    // it should have at least one datapoint to work correctly.
    aNewLogger->log( theModel->getCurrentTime() );
    aNewLogger->flush();

    // set logger policy
    aNewLogger->setLoggerPolicy( aParamList );


    return aNewLogger;
  }

  

} // namespace libecs








