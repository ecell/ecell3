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
// modify it under the terms of the GNU General Public // License as published by the Free Software Foundation; either
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
// written by Moriyoshi Koizumi <mozo@sfc.keio.ac.jp>
// E-Cell Project.
//

#ifndef __PROPERTYTYPE_HPP_
#define __PROPERTYTYPE_HPP_

#include "libecs.hpp"

namespace libecs {

class Polymorph;

class PropertyType
{
public:
    enum Code {
        _NONE      = 0,
        _INTEGER   = 1,
        _REAL      = 2,
        _STRING    = 3,
        _POLYMORPH = 4
    };

public:
    static const PropertyType& get( const String& );

    static const PropertyType& get( enum Code );

private:
    PropertyType( enum Code _code, const String& _name )
        : code( _code ), name( _name )
    {
        ; // do nothing
    }

public:
    static const PropertyType NONE;
    static const PropertyType INTEGER;
    static const PropertyType REAL;
    static const PropertyType STRING;
    static const PropertyType POLYMORPH;

    enum Code code;
    const String& name;

private:
    const PropertyType* prev;
    static const PropertyType* last;
};

template<typename T>
struct Type2PropertyType
{
    static const PropertyType& value;
};


template<typename T>
const PropertyType& Type2PropertyType<T>::value( PropertyType::NONE );

template<>
const PropertyType& Type2PropertyType<Integer>::value( PropertyType::INTEGER );

template<>
const PropertyType& Type2PropertyType<Real>::value( PropertyType::REAL );

template<>
const PropertyType& Type2PropertyType<String>::value( PropertyType::STRING );

template<>
const PropertyType& Type2PropertyType<Polymorph>::value( PropertyType::POLYMORPH );

} // namespace libesc

#endif /* __PROPERTYTYPE_HPP_ */
