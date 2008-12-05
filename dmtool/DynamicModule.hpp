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

#ifndef __DYNAMICMODULE_HPP
#define __DYNAMICMODULE_HPP

#include "dmtool/DMException.hpp"
#include "dmtool/DynamicModuleDescriptor.hpp"

/// doc needed

template< typename T >
struct SimpleAllocatorDef
{
    typedef T* (*type)();
};

enum DynamicModuleType
{
    DM_TYPE_STATIC,
    DM_TYPE_SHARED
};

class DynamicModuleInfo;

/**
  Common base class of DynamicModule and SharedDynamicModule
*/
  
template < class T, class _TAllocator = typename SimpleAllocatorDef< T >::type >
class DynamicModule
{
public:

    typedef _TAllocator DMAllocator;
    typedef const char* (DynamicModule::*FileNameGetterType)() const;
    typedef void (DynamicModule::*FinalizerType)();

public:        

    DynamicModule( DynamicModuleDescriptor const& desc,
                   enum DynamicModuleType aType,
                   FileNameGetterType aFileNameGetter = 0,
                   FinalizerType aFinalizer = 0 )
        : theDescriptor( desc ), theFileNameGetter( aFileNameGetter ),
          theFinalizer( aFinalizer )
    {
        desc.moduleInitializer();
    }

    ~DynamicModule()
    {
        theDescriptor.moduleFinalizer();
        if ( theFinalizer )
            ( this->*theFinalizer )();
    }

    const DynamicModuleDescriptor& getDescriptor()
    {
        return theDescriptor;
    }
 
    const char* getModuleName() const
    {
        return theDescriptor.moduleName;
    }

    const char* getFileName() const
    {
        return theFileNameGetter ? (this->*theFileNameGetter)(): 0;
    }

    DMAllocator getAllocator() const
    {
        return reinterpret_cast< DMAllocator >( theDescriptor.allocator );
    }

    const DynamicModuleInfo* getInfo() const
    {
        return theDescriptor.infoLoader ? (*theDescriptor.infoLoader)(): 0;
    }

    enum DynamicModuleType getType() const
    {
        return theType;
    }

protected:

    DynamicModuleDescriptor const& theDescriptor;

    FileNameGetterType theFileNameGetter;

    FinalizerType theFinalizer;

    enum DynamicModuleType theType;
};


/**
   StaticDynamicModule is a class statically linked to the binary
   that utilizes ModuleMaker
 */
template < class T, class DMAllocator = typename SimpleAllocatorDef< T >::type >
class StaticDynamicModule : public DynamicModule< T, DMAllocator >
{
public:

    typedef DynamicModule< T, DMAllocator > Base;    

public:

    StaticDynamicModule( DynamicModuleDescriptor const& desc )
        : DynamicModule< T, DMAllocator >( desc, DM_TYPE_STATIC )
    {
        ; // do nothing
    }
};

#endif /* __DYNAMICMODULE_HPP */
