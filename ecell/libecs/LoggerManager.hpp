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


#ifndef __LOGGERMANAGER_HPP
#define __LOGGERMANAGER_HPP

#include <map>
#include <boost/shared_ptr.hpp>
#include <boost/noncopyable.hpp>
#include "libecs.hpp"
#include "Happening.hpp"
#include "FullPN.hpp"
#include "Logger.hpp"
#include "DataPoint.hpp"

/**
   @addtogroup logging
 */
/** @{ */
/** @file */

namespace libecs
{
class Model;
class PropertySlot;
class Entity;
/**
   LoggerManager creates and administrates Loggers in a model.

   This class creates, holds in a map which associates FullPN with a Logger,
   and responds to requests to Loggers.

   @see FullPN
   @see Logger

*/
class LIBECS_API LoggerManager: private boost::noncopyable
{
public:
    typedef boost::shared_ptr< Logger > Handle;

private:
    typedef Happening< Handle, Logger::DataPoint > LoggingEventDispatcher;
    typedef std::pair< const PropertySlot*, LoggingEventDispatcher > Entry;
    typedef std::map< const String, Entry > PNToDispatcherMap;
    typedef std::map< const Entity*, PNToDispatcherMap > DispatcherMap;

public:
    LoggerManager( Model* model );

    ~LoggerManager();

    void add( const FullPN& fullPN, Handle logger );
    void remove( const FullPN& fullPN, Handle logger );

    void log( TimeParam currentTime, const Entity* ent ) const;

private:
    DispatcherMap     dispatchers_;
    Model*            model_;
};

/** @} */

} // namespace libecs

#endif /* __LOGGERMANAGER_HPP */
