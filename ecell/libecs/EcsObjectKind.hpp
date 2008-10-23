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

class EcsObjectKind
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
    static const EcsObjectKind& get( const String& );

    static const EcsObjectKind& get( enum Code );

    static const EcsObjectKind& fromEntityType( const EntityType& );

    operator const String&() const
    {
        return name;
    }

    operator const char* const() const
    {
        return name.c_str();
    }

private:
    EcsObjectKind( enum Code _code, const String& _name )
        : code( _code ), name( _name )
    {
        ; // do nothing
    }

public:
    static const EcsObjectKind NONE;
    static const EcsObjectKind STEPPER;
    static const EcsObjectKind VARIABLE;
    static const EcsObjectKind PROCESS;
    static const EcsObjectKind SYSTEM;

    enum Code code;
    const String& name;

private:
    const EcsObjectKind* prev;
    static const EcsObjectKind* last;
};

template<typename T>
struct Type2EcsObjectKind
{
    static const EcsObjectKind& value;
};

template<typename T>
const EcsObjectKind& Type2EcsObjectKind<T>::value(
        EcsObjectKind::NONE );

} // namespace libesc

#endif /* __PROPERTIEDCLASSKIND_HPP_ */
