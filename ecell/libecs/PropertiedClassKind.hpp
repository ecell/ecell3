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

#ifndef __PROPERTIEDCLASSKIND_HPP_
#define __PROPERTIEDCLASSKIND_HPP_

#include "libecs.hpp"

namespace libecs {

class Stepper;
class Process;
class Variable;
class System;
class EntityType;

class PropertiedClassKind
{
public:
    enum Code {
        _NONE      = 0,
        _STEPPER   = 1,
        _VARIABLE  = 2,
        _PROCESS   = 3,
        _SYSTEM    = 4
    };

public:
    static const PropertiedClassKind& get( const String& );

    static const PropertiedClassKind& get( enum Code );

    static const PropertiedClassKind& fromEntityType( const EntityType& );

    operator const String&() const
    {
        return name;
    }

    operator const char* const() const
    {
        return name.c_str();
    }

private:
    PropertiedClassKind( enum Code _code, const String& _name )
        : code( _code ), name( _name )
    {
        ; // do nothing
    }

public:
    static const PropertiedClassKind NONE;
    static const PropertiedClassKind STEPPER;
    static const PropertiedClassKind VARIABLE;
    static const PropertiedClassKind PROCESS;
    static const PropertiedClassKind SYSTEM;

    enum Code code;
    const String& name;

private:
    const PropertiedClassKind* prev;
    static const PropertiedClassKind* last;
};

template<typename T>
struct Type2PropertiedClassKind
{
    static const PropertiedClassKind& value;
};

template<typename T>
const PropertiedClassKind& Type2PropertiedClassKind<T>::value(
        PropertiedClassKind::NONE );

} // namespace libesc

#endif /* __PROPERTIEDCLASSKIND_HPP_ */
