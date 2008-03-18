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

#ifndef __FULLPN_HPP
#define __FULLPN_HPP

#include "libecs.hpp"
#include "FullID.hpp"

/** @addtogroup identifier The FullID, FullPN and SystemPath.
 The FullID, FullPN and SystemPath.
 

 @ingroup libecs
 @{
 */

/** @file */


namespace libecs {

class FullPN
{
public:
    FullPN( const EntityType& type,
            const SystemPath& systempath,
            const String& id,
            const String& propertyname )
        : fullID_( type, systempath, id ),
          propertyName_( propertyname )
    {
        ; // do nothing
    }

    FullPN( const FullID& fullid, const String& propertyname )
        : fullID_( fullid ),
          propertyName_( propertyname )
    {
        ; // do nothing
    }

    FullPN( const FullPN& fullpn )
        : fullID_( fullpn.getFullID() ),
          propertyName_( fullpn.getPropertyName() )
    {
        ; // do nothing
    }

    ~FullPN()
    {
        ; // do nothing
    }

    const FullID& getFullID() const
    {
        return fullID_;
    }

    const EntityType  getEntityType() const
    {
        return getFullID().getEntityType();
    }

    const SystemPath& getSystemPath() const
    {
        return getFullID().getSystemPath();
    }

    const String& getID() const
    {
        return getFullID().getID();
    }

    const String& getPropertyName() const
    {
        return propertyName_;
    }

    LIBECS_API static FullPN parse( const String& that );

    LIBECS_API const String asString() const;

    bool operator<( const FullPN& rhs ) const
    {
        if ( getFullID() != rhs.getFullID() )
        {
            return getFullID() < rhs.getFullID();
        }

        return getPropertyName() < rhs.getPropertyName();
    }

    bool operator==( const FullPN& rhs ) const
    {
        return getFullID() == rhs.getFullID() &&
               getPropertyName() == rhs.getPropertyName();
    }

    bool operator!=( const FullPN& rhs ) const
    {
        return ! operator==( rhs );
    }

private:
    FullID fullID_;
    String propertyName_;
};

} // namespace libecs

/** @} */ // identifier module

#endif // __FULLPN_HPP

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
