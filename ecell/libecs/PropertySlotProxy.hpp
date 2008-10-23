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

#include "libecs.hpp"
#include "Converters.hpp"
#include "Polymorph.hpp"

/** @addtogroup property
  
@ingroup libecs
@{
*/

/** @file */

#include "PropertySlot.hpp"

#ifndef __LIBECS_PROPERTYSLOTPROXYPROXY_DEFINED
#define __LIBECS_PROPERTYSLOTPROXYPROXY_DEFINED
namespace libecs {

class EcsObject;

class PropertySlotProxy
{
public:
    PropertySlotProxy( EcsObject* anObject,
            const PropertySlot* aSlot )
        : obj_( anObject ), slot_( aSlot )
    {
        ; // do nothing
    }

    template < typename Type > void set( typename Param<Type>::type aValue )
    {
        slot_->set<Type>( *obj_, aValue );
    }

    template < typename Type > Type get() const
    {
        return slot_->get<Type>( *obj_ );
    }

    void load( Param<Polymorph>::type aValue )
    {
        slot_->load( *obj_, aValue );
    }

    const Polymorph save() const
    {
        return slot_->save( *obj_ );
    }

protected:
    EcsObject* obj_;
    const PropertySlot* slot_;
};

} // namespace libecs
#endif /* __LIBECS_PROPERTYSLOTPROXY_DEFINED */

/** @}*/
