//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2009 Keio University
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
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//
#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include "System.hpp"
#include "FullID.hpp"

#include "Entity.hpp"
#include "Model.hpp"

namespace libecs
{

LIBECS_DM_INIT_STATIC( Entity, Entity );

Entity::Entity()
    : theSuperSystem( NULLPTR ),
      theID( "" ),
      theName( "" ),
      theLoggerMap( NULLPTR )
{

}


Entity::~Entity()
{
    ; // do nothing
}

const FullID Entity::getFullID() const
{
    return FullID( getEntityType(), getSystemPath(), getID() );
}

const SystemPath Entity::getSystemPath() const
{
    System* aSystem( getSuperSystem() );

    if ( !aSystem )
    {
        THROW_EXCEPTION( IllegalOperation, "no system is associated" );
    }

    if ( aSystem == this )
    {
        return SystemPath();
    }

    SystemPath aSystemPath( aSystem->getSystemPath() );
    aSystemPath.push_back( aSystem->getID() );
    return aSystemPath;
}

LoggerBroker::LoggersPerFullID
Entity::getLoggers() const
{
    LoggerBroker::PerFullIDMap* aLoggerMap(
        theLoggerMap ? theLoggerMap:
            &getModel()->getLoggerBroker().theEmptyPerFullIDMap );
    return LoggerBroker::LoggersPerFullID(
        LoggerBroker::PerFullIDLoggerIterator(
            aLoggerMap->begin(),
            SelectSecond< LoggerBroker::PerFullIDMap::value_type >() ),
        LoggerBroker::PerFullIDLoggerIterator(
            aLoggerMap->end(),
            SelectSecond< LoggerBroker::PerFullIDMap::value_type >() )
    );
    // return getModel()->getLoggerBroker().getLoggersByFullID( getFullID() );
}


String Entity::asString() const
{
    return getPropertyInterface().getClassName() + "["
            + ( theSuperSystem ? getFullID().asString():
                                 "unbound: " + getID() ) + "]";
}


} // namespace libecs
