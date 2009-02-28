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

#ifndef __SHAREDDYNAMICMODULE_HPP
#define __SHAREDDYNAMICMODULE_HPP

#include "dmtool/DynamicModule.hpp"
#include <ltdl.h>

/**
   SharedDynamicModule loads a class from a shared object file
   and instantiate objects of the loaded class.
   It opens and loads a shared object(.so) file into memory
   when constructed. It closes the file when deleted.

   The shared object must have the following (in "C" signature):
     - __dm_info                      - a DynamicModuleDescriptor
                                        which holds the module information.
 */

template < class T, class DMAllocator = typename SimpleAllocatorDef< T >::type >
class SharedDynamicModule : public DynamicModule< T, DMAllocator >
{
public:

    typedef DynamicModule< T, DMAllocator > Base;    

public:

    SharedDynamicModule( DynamicModuleDescriptor const& desc,
                         const std::string& fileName,
                         lt_dlhandle handle )
        : DynamicModule< T, DMAllocator >(
            desc, DM_TYPE_SHARED,
            reinterpret_cast< typename Base::FileNameGetterType >(
                &SharedDynamicModule::getFileName ),
            reinterpret_cast< typename Base::FinalizerType >(
                &SharedDynamicModule::finalize ) )
    {
        ; // do nothing
    }

private:

    void finalize()
    {
        if( this->theHandle )
        {
            lt_dlclose( this->theHandle );
            this->theHandle = 0;
        }
    }


    const char* getFileName() const
    {
        return theFileName.c_str();
    }

private:

    lt_dlhandle theHandle;
    std::string theFileName;
};

#endif /* __SHAREDDYNAMICMODULE_HPP */
