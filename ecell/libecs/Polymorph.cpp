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

#include <typeinfo>

#include "Util.hpp"
#include "Exceptions.hpp"

#include "Polymorph.hpp"

namespace libecs
{

PolymorphValue::~PolymorphValue()
{
    ; // do nothing
}

PolymorphNoneValue::~PolymorphNoneValue()
{
    ; // do nothing
}

const PolymorphVector PolymorphNoneValue::asPolymorphVector() const
{
    return PolymorphVector();
}

const String PolymorphNoneValue::asString() const
{
    return String();
}

const Polymorph::Type Polymorph::getType() const
{
    if ( typeid( *value_ ) == typeid( ConcretePolymorphValue<Real> ) )
    {
        return REAL;
    }
    else if ( typeid( *value_ ) == typeid( ConcretePolymorphValue<Integer> ) )
    {
        return INTEGER;
    }
    else if ( typeid( *value_ ) == typeid( ConcretePolymorphValue<String> ) )
    {
        return STRING;
    }
    else if ( typeid( *value_ ) ==
              typeid( ConcretePolymorphValue<PolymorphVector> ) )
    {
        return POLYMORPH_VECTOR;
    }
    else if ( typeid( *value_ ) == typeid( PolymorphNoneValue ) )
    {
        return NONE;
    }

    NEVER_GET_HERE;
}


void Polymorph::changeType( const Type aType )
{
    PolymorphValue* pval( NULLPTR );

    switch ( aType )
    {
    case REAL:
        pval =
            new ConcretePolymorphValue<Real>( value_->asReal() );
        break;
    case INTEGER:
        pval =
            new ConcretePolymorphValue<Integer>( value_->asInteger() );
        break;
    case STRING:
        pval =
            new ConcretePolymorphValue<String>( value_->asString() );
        break;
    case POLYMORPH_VECTOR:
        pval =
            new ConcretePolymorphValue<PolymorphVector>
        ( value_->asPolymorphVector() );
        break;
    case NONE:
        pval = new PolymorphNoneValue();
        break;
    default:
        NEVER_GET_HERE;
    }

    delete value_;
    value_ = pval;
}

} // namespace libecs
