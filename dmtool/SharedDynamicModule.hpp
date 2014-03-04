//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2014 Keio University
//       Copyright (C) 2008-2014 RIKEN
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

template < class T >
class SharedDynamicModule : public StaticDynamicModule< T >
{
public:

    typedef StaticDynamicModule< T > Base;

public:

    SharedDynamicModule( DynamicModuleDescriptor const& desc,
                         const std::string& fileName,
                         lt_dlhandle handle )
        : Base( DM_TYPE_SHARED, desc ),
          theFileName( fileName ),
          theHandle( handle )
    {
        ; // do nothing
    }

private:

    virtual ~SharedDynamicModule()
    {
        // make sure that the finalizer is called before
        // closing dynamic library
        ( *Base::theDescriptor->moduleFinalizer )();
        Base::theDescriptor = 0;
        if( this->theHandle )
        {
            lt_dlclose( this->theHandle );
            this->theHandle = 0;
        }
    }


    virtual const char* getFileName() const
    {
        return theFileName.c_str();
    }

private:

    lt_dlhandle theHandle;
    std::string theFileName;
};

#endif /* __SHAREDDYNAMICMODULE_HPP */
