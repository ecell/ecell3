//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2015 Keio University
//       Copyright (C) 2008-2015 RIKEN
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
// written by Masayuki Okayama <smash@e-cell.org>,
// E-Cell Project.
//

#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include <algorithm>
#include <functional>

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

LoggerBroker::LoggerBroker( Model const& aModel )
    : theModel( aModel )
{
    ; // do nothing
}

LoggerBroker::~LoggerBroker()
{
    std::for_each( begin(), end(),
            ComposeUnary(
                DeletePtr< Logger >(),
                SelectSecond< iterator::value_type >() ) );
}

void LoggerBroker::flush()
{
    std::for_each( begin(), end(),
            ComposeUnary(
                std::mem_fun( &Logger::flush ),
                SelectSecond< iterator::value_type >() ) );
}

Logger* LoggerBroker::getLogger( FullPN const& aFullPN ) const
{
    LoggerMap::const_iterator anOuterIter(
        theLoggerMap.find( aFullPN.getFullID() ) );

    if( anOuterIter == theLoggerMap.end() )
    {
        THROW_EXCEPTION( NotFound, "logger [" + aFullPN.asString() 
                                   + "] not found" );
    }

    PerFullIDMap::const_iterator anInnerIter(
        anOuterIter->second.find( aFullPN.getPropertyName() ) );

    if( anInnerIter == anOuterIter->second.end() )
    {
        THROW_EXCEPTION( NotFound, "logger [" + aFullPN.asString() 
                                   + "] not found" );
    }

    return anInnerIter->second;
}

Logger* LoggerBroker::createLogger( FullPN const& aFullPN,
                                    Logger::Policy const& aPolicy )
{
    LoggerMap::const_iterator anOuterIter(
        theLoggerMap.find( aFullPN.getFullID() ) );

    if( anOuterIter != theLoggerMap.end() )
    {
        PerFullIDMap::const_iterator anInnerIter(
            anOuterIter->second.find( aFullPN.getPropertyName() ) );

        if( anInnerIter != anOuterIter->second.end() )
        {
            THROW_EXCEPTION( AlreadyExist, "Logger [" + aFullPN.asString()
                                           + "] already exists" );
        }
    }

    Entity* const anEntity( theModel.getEntity( aFullPN.getFullID() ) );

    const String aPropertyName( aFullPN.getPropertyName() );

    PropertySlotProxy* const aPropertySlotProxy(
        anEntity->createPropertySlotProxy( aPropertyName ) );

    LoggerAdapter* const aLoggerAdapter(
        new PropertySlotProxyLoggerAdapter( aPropertySlotProxy ) );

    Logger* const aNewLogger( new Logger( aLoggerAdapter, aPolicy ) );

    std::pair< LoggerMap::iterator, bool > anInnerMap(
        theLoggerMap.insert(
            LoggerMap::value_type( aFullPN.getFullID(), PerFullIDMap() ) ) );

    try
    {
        ( (*anInnerMap.first).second )[ aFullPN.getPropertyName() ] = aNewLogger;
        if ( anInnerMap.second )
            anEntity->setLoggerMap( &( *anInnerMap.first ).second );

        // it should have at least one datapoint to work correctly.
        aNewLogger->log( theModel.getCurrentTime() );
        aNewLogger->flush();
    }
    catch ( std::exception const& )
    {
        if ( anInnerMap.second )
        {
            anEntity->setLoggerMap( 0 );
            theLoggerMap.erase( anInnerMap.first );
        }
        delete aNewLogger;
        throw;
    } 

    return aNewLogger;
}

void LoggerBroker::removeLogger( FullPN const& aFullPN )
{
    LoggerMap::iterator anOuterIter(
        theLoggerMap.find( aFullPN.getFullID() ) );

    if( anOuterIter == theLoggerMap.end() )
    {
        THROW_EXCEPTION( NotFound, "Logger [" + aFullPN.asString() 
                                   + "] not found" );
    }

    PerFullIDMap::iterator anInnerIter(
        anOuterIter->second.find( aFullPN.getPropertyName() ) );

    if( anInnerIter == anOuterIter->second.end() )
    {
        THROW_EXCEPTION( NotFound, "Logger [" + aFullPN.asString() 
                                   + "] not found" );
    }

    anOuterIter->second.erase( anInnerIter );
    if ( anOuterIter->second.empty() )
    {
        Entity* const anEntity( theModel.getEntity( aFullPN.getFullID() ) );
        anEntity->setLoggerMap( 0 );
        theLoggerMap.erase( anOuterIter );
    }

    delete (*anInnerIter).second;
}

LoggerBroker::LoggersPerFullID
LoggerBroker::getLoggersByFullID( FullID const& aFullID )
{
    LoggerMap::iterator anOuterIter( theLoggerMap.find( aFullID ) );
    PerFullIDMap& anInnerMap(
        anOuterIter == theLoggerMap.end() ? theEmptyPerFullIDMap:
                                            anOuterIter->second );

    return LoggersPerFullID(
        PerFullIDLoggerIterator(
            anInnerMap.begin(),
            SelectSecond< LoggerMap::value_type::second_type::value_type >() ),
        PerFullIDLoggerIterator(
            anInnerMap.end(),
            SelectSecond< LoggerMap::value_type::second_type::value_type >() )
    );
}

LoggerBroker::ConstLoggersPerFullID
LoggerBroker::getLoggersByFullID( FullID const& aFullID ) const
{
    LoggerMap::const_iterator anOuterIter( theLoggerMap.find( aFullID ) );
    PerFullIDMap const& anInnerMap(
        anOuterIter == theLoggerMap.end() ? theEmptyPerFullIDMap:
                                            anOuterIter->second );

    return ConstLoggersPerFullID(
        PerFullIDLoggerConstIterator(
            anInnerMap.begin(),
            SelectSecond< LoggerMap::value_type::second_type::value_type >() ),
        PerFullIDLoggerConstIterator(
            anInnerMap.end(),
            SelectSecond< LoggerMap::value_type::second_type::value_type >() )
    );
}

void LoggerBroker::removeLoggersByFullID( FullID const& aFullID )
{
    LoggerMap::iterator anOuterIter( theLoggerMap.find( aFullID ) );

    if( anOuterIter == theLoggerMap.end() )
    {
        THROW_EXCEPTION( NotFound, "Logger for [" + aFullID.asString() 
                                   + "] not found" );
    }

    std::for_each( (*anOuterIter).second.begin(),
                   (*anOuterIter).second.end(),
                   ComposeUnary(
                       DeletePtr< Logger >(),
                       SelectSecond< PerFullIDMap::value_type >() ) );
    theLoggerMap.erase( anOuterIter );
}


} // namespace libecs
