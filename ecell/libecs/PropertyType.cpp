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
#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include <assert.h>

#include "PropertyType.hpp"


namespace libecs {

const PropertyType* PropertyType::last( 0 );
const PropertyType PropertyType::      NONE( _NONE     , "None" );
const PropertyType PropertyType::   INTEGER( _INTEGER  , "Integer" );
const PropertyType PropertyType::      REAL( _REAL     , "Real" );
const PropertyType PropertyType::    STRING( _STRING   , "String" );
const PropertyType PropertyType:: POLYMORPH( _POLYMORPH, "Polymorph" );

const PropertyType& PropertyType::get( const String& name )
{
    for ( const PropertyType* item = last; item; item = item->prev )
    {
        if ( item->name == name )
        {
            return *item;
        }
    }
}

const PropertyType& PropertyType::get( enum Code code )
{
    for ( const PropertyType* item = last; item; item = item->prev )
    {
        if ( item->code == code )
        {
            return *item;
        }
    }
}

} // namespace libecs
