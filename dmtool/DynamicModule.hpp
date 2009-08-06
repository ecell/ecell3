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

#ifndef __DYNAMICMODULE_HPP
#define __DYNAMICMODULE_HPP

#include "dmtool/DMException.hpp"
#include "dmtool/DynamicModuleDescriptor.hpp"

/// doc needed

enum DynamicModuleType
{
    DM_TYPE_BUILTIN,
    DM_TYPE_SHARED,
    DM_TYPE_DYNAMIC
};

class DynamicModuleInfo;

/**
  Common base class of DynamicModule and SharedDynamicModule
*/
  
template < class T >
class DynamicModule
{
public:        

    DynamicModule( enum DynamicModuleType aType ): theType( aType )
    {
    }

    virtual ~DynamicModule() {}
 
    virtual const char* getModuleName() const = 0;

    virtual const char* getFileName() const = 0;

    virtual T* createInstance() const = 0;

    virtual const DynamicModuleInfo* getInfo() const = 0;

    enum DynamicModuleType getType() const
    {
        return theType;
    }

protected:

    enum DynamicModuleType theType;
};


template < class T >
class StaticDynamicModule: public DynamicModule< T >
{
public:

    typedef DynamicModule< T > Base;

public:        

    StaticDynamicModule( enum DynamicModuleType aType,
                   DynamicModuleDescriptor const& desc )
        : Base( aType ), theDescriptor( desc )
    {
        theDescriptor.moduleInitializer();
    }

    virtual ~StaticDynamicModule()
    {
        theDescriptor.moduleFinalizer();
    }

    const DynamicModuleDescriptor& getDescriptor()
    {
        return theDescriptor;
    }
 
    virtual const char* getModuleName() const
    {
        return theDescriptor.moduleName;
    }

    virtual T* createInstance() const
    {
        return reinterpret_cast< T* >( ( *theDescriptor.allocator )() );
    }

    virtual const DynamicModuleInfo* getInfo() const
    {
        return ( *theDescriptor.infoLoader )();
    }

protected:

    DynamicModuleDescriptor const& theDescriptor;
};


/**
   BuiltinDynamicModule is a class statically linked to the binary
   that utilizes ModuleMaker
 */
template < class T >
class BuiltinDynamicModule : public StaticDynamicModule< T >
{
public:

    typedef StaticDynamicModule< T > Base;

public:

    BuiltinDynamicModule( DynamicModuleDescriptor const& desc )
        : Base( DM_TYPE_BUILTIN, desc )
    {
        ; // do nothing
    }

    virtual const char* getFileName() const
    {
        return "<builtin>";
    }
};

#endif /* __DYNAMICMODULE_HPP */
