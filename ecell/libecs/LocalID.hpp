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

#ifndef __LOCALID_HPP
#define __LOCALID_HPP

#include <string.h>
#include "libecs.hpp"
#include "EntityType.hpp"

/** @addtogroup identifier The FullID, FullPN and SystemPath.
 The FullID, FullPN and SystemPath.
 

 @ingroup libecs
 @{
 */

/** @file */


namespace libecs {

/**
   LocalID is an identifier that is unique within a System.
 */
class LocalID
{
public:
    LocalID( const EntityType& type,
             const String& id )
        : entityType_( &type ), id_( id )
    {
        ; // do nothing
    }

    ~LocalID() {}

    const EntityType  getEntityType() const
    {
        return *entityType_;
    }

    const String& getID() const
    {
        return id_;
    }

    bool operator<( const LocalID& rhs ) const
    {
        // first look at the EntityType
        if ( getEntityType() != rhs.getEntityType() )
        {
            return getEntityType() < rhs.getEntityType();
        }

        // finally compare the ID strings
        return getID() < rhs.getID();
    }

    bool operator==( const LocalID& rhs ) const
    {
        return getEntityType() == rhs.getEntityType() &&
                getID() == rhs.getID();
    }

    bool operator!=( const LocalID& rhs ) const
    {
        return ! operator==( rhs );
    }

private:
    const EntityType* entityType_;
    String id_;
};

/** @} */ // identifier module

} // namespace libecs

#endif // __LOCALID_HPP

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
