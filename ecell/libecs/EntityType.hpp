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
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#ifndef __ENTITYTYPE_HPP
#define __ENTITYTYPE_HPP

#include "libecs.hpp"
#include "Exceptions.hpp"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>

namespace libecs
{

/** @addtogroup identifier
 *
 *@{
 */

/** @file */

class EcsObjectKind;

/**
  A decorated enumeration type that lists the types of entities
 */
struct LIBECS_API EntityType
{
public:
    enum Code {
        _NONE      = 0,
        _ENTITY    = 1,
        _VARIABLE  = 2,
        _PROCESS   = 3,
        _SYSTEM    = 4
    };

public:
    bool operator<( const EntityType& rhs ) const
    {
        return code < rhs.code;
    }

    bool operator==( const EntityType& rhs ) const
    {
        return code == rhs.code;
    }

    bool operator!=( const EntityType& rhs ) const
    {
        return code != rhs.code;
    }

    template<typename TcharRange_>
    static const EntityType& get( const TcharRange_& name )
    {
        for ( const EntityType* item = last; item; item = item->prev )
        {
            if ( boost::equals( item->name, name ) )
            {
                return *item;
            }
        }

        THROW_EXCEPTION( ValueError,
            String( "no EntityType named " )
            + String( boost::begin( name ), boost::end( name ) ) );
    }


    static const EntityType& get( enum Code );

    static const EntityType& fromEcsObjectKind( const EcsObjectKind& );

    operator const String&() const
    {
        return name;
    }

    operator const char* const() const
    {
        return name.c_str();
    }
private:
    EntityType( const Code _code, const String& _name )
        : code( _code ), name( _name ), prev( last )
    {
        last = this;
    }

public:
    static const EntityType NONE;
    static const EntityType ENTITY;
    static const EntityType VARIABLE;
    static const EntityType PROCESS;
    static const EntityType SYSTEM;

    enum Code code;
    const String name;
private:
    const EntityType* prev;
    static const EntityType* last;
};

/*@}*/

} // namespace libecs


#endif /* __ENTITYTYPE_HPP */
