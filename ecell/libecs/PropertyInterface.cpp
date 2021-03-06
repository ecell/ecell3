//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2016 Keio University
//       Copyright (C) 2008-2016 RIKEN
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
// modified by Masayuki Okayama <smash@e-cell.org>,
// E-Cell Project.
//

#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include "PropertyInterface.hpp"

namespace libecs
{

void PropertyInterfaceBase::throwNoSlot( String const& aPropertyName ) const
{
    THROW_EXCEPTION( NoSlot,
                     getClassName() + 
                     ": has no such property [" +
                     aPropertyName + String( "]" ) );
}

void PropertyInterfaceBase::throwNoSlot( EcsObject const& obj, String const& aPropertyName ) const
{
    THROW_EXCEPTION_ECSOBJECT( NoSlot,
                     getClassName() + 
                     ": has no such property [" +
                     aPropertyName + String( "]" ),
                     &obj );
}

void PropertyInterfaceBase::throwNotLoadable( EcsObject const& obj, String const& aPropertyName ) const
{
    THROW_EXCEPTION_ECSOBJECT( NoSlot,
                    getClassName() + 
                    ": property [" +
                    aPropertyName + String( "] is not loadable" ),
                    &obj );
}


void PropertyInterfaceBase::throwNotSavable( EcsObject const& obj, String const& aPropertyName ) const
{
    THROW_EXCEPTION_ECSOBJECT( NoSlot,
                     getClassName() + 
                     ": property [" +
                     aPropertyName + String( "] is not savable" ),
                     &obj );
}

} // namespace libecs
