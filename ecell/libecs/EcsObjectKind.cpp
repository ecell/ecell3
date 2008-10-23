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

#include "EcsObjectKind.hpp"
#include "EntityType.hpp"
#include "Exceptions.hpp"

namespace libecs {

const EcsObjectKind* EcsObjectKind::last( 0 );
const EcsObjectKind EcsObjectKind::      NONE( _NONE    , "None" );
const EcsObjectKind EcsObjectKind::   STEPPER( _STEPPER , "Stepper" );
const EcsObjectKind EcsObjectKind::  VARIABLE( _VARIABLE, "Variable" );
const EcsObjectKind EcsObjectKind::   PROCESS( _PROCESS,  "Process" );
const EcsObjectKind EcsObjectKind::    SYSTEM( _SYSTEM,   "System" );

const EcsObjectKind& EcsObjectKind::get( const String& name )
{
    for ( const EcsObjectKind* item = last; item; item = item->prev )
    {
        if ( item->name == name )
        {
            return *item;
        }
    }
}

const EcsObjectKind& EcsObjectKind::get( enum Code code )
{
    for ( const EcsObjectKind* item = last; item; item = item->prev )
    {
        if ( item->code == code )
        {
            return *item;
        }
    }
}

const EcsObjectKind&
EcsObjectKind::fromEntityType( const EntityType& et )
{
    switch ( et.code )
    {
    case EntityType::_PROCESS:
        return EcsObjectKind::PROCESS;
    case EntityType::_VARIABLE:
        return EcsObjectKind::VARIABLE;
    case EntityType::_SYSTEM:
        return EcsObjectKind::SYSTEM;
    }
    THROW_EXCEPTION( ValueError,
        String( "no EcsObjectKind counterpart for " )
        + static_cast< const String& >( et ) );
}

template<>
const EcsObjectKind& Type2EcsObjectKind<Stepper>::value(
        EcsObjectKind::STEPPER );

template<>
const EcsObjectKind& Type2EcsObjectKind<Variable>::value(
        EcsObjectKind::VARIABLE );

template<>
const EcsObjectKind& Type2EcsObjectKind<Process>::value(
        EcsObjectKind::PROCESS );

template<>
const EcsObjectKind& Type2EcsObjectKind<System>::value(
        EcsObjectKind::SYSTEM );


} // namespace libecs
