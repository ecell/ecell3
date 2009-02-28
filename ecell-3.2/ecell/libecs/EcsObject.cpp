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
// modified by Masayuki Okayama <smash@e-cell.org>,
// E-Cell Project.
//

#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include <boost/format.hpp>
#include <boost/format/group.hpp>

#include "PropertyInterface.hpp"
#include "Exceptions.hpp"

#include "EcsObject.hpp"

namespace libecs
{


///////////////////////////// EcsObject
const PropertyAttributes EcsObject::
defaultGetPropertyAttributes( StringCref aPropertyName ) const
{
    THROW_EXCEPTION( NoSlot, 
                     asString() + ": No property slot ["
                     + aPropertyName + "].    Get property attributes failed." );
}

const StringVector
EcsObject::defaultGetPropertyList() const
{
    return StringVector();
}

void EcsObject::defaultSetProperty( StringCref aPropertyName, 
                                                                                    PolymorphCref aValue )
{
    THROW_EXCEPTION( NoSlot,
                     asString() + ": No property slot ["
                     + aPropertyName + "].    Set property failed." );
}

const Polymorph 
EcsObject::defaultGetProperty( StringCref aPropertyName ) const
{
    THROW_EXCEPTION( NoSlot, 
                     asString() + ": No property slot ["
                     + aPropertyName + "].    Get property failed." );
}


void EcsObject::throwNotSetable()
{
    THROW_EXCEPTION( AttributeError, "Not setable." );
}


void EcsObject::throwNotGetable()
{
    THROW_EXCEPTION( AttributeError, "Not getable." );
}


StringCref EcsObject::getClassName() const
{
    return getPropertyInterface().getClassName();
}

String EcsObject::asString() const
{
    return ( boost::format( "%s[#%p]" ) % boost::io::group(
            getPropertyInterface().getClassName(), this ) ).str();
}

#define NULLGETSET_SPECIALIZATION_DEF( TYPE )\
template <> void EcsObject::nullSet<TYPE>( Param<TYPE>::type )\
{\
    throwNotSetable();\
}\
template <> const TYPE EcsObject::nullGet<TYPE>() const\
{\
    throwNotGetable();\
    return TYPE(); \
} //

NULLGETSET_SPECIALIZATION_DEF( Real );
NULLGETSET_SPECIALIZATION_DEF( Integer );
NULLGETSET_SPECIALIZATION_DEF( String );
NULLGETSET_SPECIALIZATION_DEF( Polymorph );

} // namespace libecs
