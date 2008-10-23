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
#include "LoggerManager.hpp"


namespace libecs
{

LoggerManager::LoggerManager( Model* model )
    : model_( model )
{
}

LoggerManager::~LoggerManager()
{
}

void LoggerManager::add( const FullPN& fullPN, Handle logger )
{
    Entity* target( model_->getEntity( fullPN.getFullID() ) );
    Entry& ent( dispatchers_[ target ][ fullPN.getPropertyName() ] );
    ent.first = target->getPropertySlot( fullPN.getPropertyName() );
    ent.second.add(
            LoggingEventDispatcher::Subscription( logger, &Logger::push_back ) );
}

void LoggerManager::remove( const FullPN& fullPN, Handle logger )
{
    Entity* target( model_->getEntity( fullPN.getFullID() ) );
    dispatchers_[ target ][ fullPN.getPropertyName() ]
        .second.remove(
            LoggingEventDispatcher::Subscription( logger, &Logger::push_back ) );
}

void LoggerManager::log( TimeParam currentTime, const Entity* ent ) const
{
    const DispatcherMap::const_iterator pos( dispatchers_.find( ent ) );
    BOOST_ASSERT( pos != dispatchers_.end() );
    const PNToDispatcherMap& pnToDispatcherMap( pos->second );
    for ( PNToDispatcherMap::const_iterator i( pnToDispatcherMap.begin() );
            i != pnToDispatcherMap.end(); ++i )
    {
        ( i->second.second )(
            Logger::DataPoint( currentTime,
                i->second.first->get<Real>(
                    static_cast<const EcsObject&>( *ent ) ) ) ); 
    }
}

} // namespace libecs
