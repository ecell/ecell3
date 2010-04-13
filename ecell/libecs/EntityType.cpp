//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2010 Keio University
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
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include <assert.h>

#include "Exceptions.hpp"
#include "EntityType.hpp"

namespace libecs
{

const String EntityType::entityTypeStringOfNone( "" );

const String EntityType::entityTypeStringOfEntity( "Entity" );

const String EntityType::entityTypeStringOfProcess( "Process" );

const String EntityType::entityTypeStringOfVariable( "Variable" );

const String EntityType::entityTypeStringOfSystem( "System" );

EntityType::EntityType( String const& aTypeString )
{
    // linear search may well work here;    n < 8.

    if( aTypeString.empty() )
    {
        theType = NONE;
    }
    else if( aTypeString == entityTypeStringOfVariable )
    {
        theType = VARIABLE;
    }
    else if( aTypeString == entityTypeStringOfProcess )
    {
        theType = PROCESS;
    }
    else if( aTypeString == entityTypeStringOfSystem )
    {
        theType = SYSTEM;
    }
    else if( aTypeString == entityTypeStringOfEntity )
    {
        theType = ENTITY;
    }
    else
    {
        THROW_EXCEPTION( InvalidEntityType,
                         "cannot convert the typestring [" + aTypeString
                         + "] to EntityType" );
    }
}

String const& EntityType::asString() const
{
    switch( theType )
    {
    case NONE:
        return entityTypeStringOfNone;
    case VARIABLE:
        return entityTypeStringOfVariable;
    case PROCESS:
        return entityTypeStringOfProcess;
    case SYSTEM:
        return entityTypeStringOfSystem;
    case ENTITY:
        return entityTypeStringOfEntity;
    default:
        THROW_EXCEPTION( InvalidEntityType,
                         "unexpected EntityType::Type" );
    }
}

} // namespace libecs

