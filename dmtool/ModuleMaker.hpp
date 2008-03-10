//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2007 Keio University
//       Copyright (C) 2005-2007 The Molecular Sciences Institute
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


#ifndef __MODULEMAKER_HPP
#define __MODULEMAKER_HPP

#include <iostream>
#include <map>
#include <string>
#include <assert.h>
#include "ltdl.h"
#include "DynamicModule.hpp"

/// doc needed

#define DynamicModuleEntry( T )\
addClass( new Module( std::string( #T ) ) );

/**
   A base class for ModuleMakers
 */

class ModuleMaker
{

public:

    ModuleMaker()
    {
        ; // do nothing
    }

    virtual ~ModuleMaker()
    {
        ; // do nothing
    }

    static void setSearchPath( const std::string& path )
    {
        int error = lt_dlsetsearchpath( path.c_str() );
        if ( error != 0 )
        {
            throw DMException( lt_dlerror() );
        }
    }

    static const std::string getSearchPath()
    {
        const char* aSearchPath( lt_dlgetsearchpath() );
        if ( aSearchPath == 0 )
        {
            return "";
        }
        else
        {
            return aSearchPath;
        }
    }

    /**
      Initializes the dynamic module facility.
      Applications that use this library must call this function
      prior to any operation involved with the facility.
      @return true on error, false otherwise.
     */
    static bool initialize()
    {
        return lt_dlinit() > 0 ? true: false;
    }

    /**
      Finalizes the dynamic module facility.
      Applications that use this library must call this function when
      the facility is no longer necessary so that allocated resources
      can be reclaimed.
      @return true on error, false otherwise.
     */
    static void finalize()
    {
        lt_dlexit();
    }
};


/**
  StaticModuleMaker is used to instantiate
  various subclasses of certain template parameter class T.
*/

template<class T_>
class StaticModuleMaker: public ModuleMaker
{
public:
    typedef T_ ObjectType;
    typedef DynamicModuleBase< T_ > Module;
    typedef std::map<const std::string, Module*> ModuleMap;
    typedef typename ModuleMap::iterator ModuleMapIterator;

public:
    StaticModuleMaker()
    {
        ; // do nothing
    }

    virtual ~StaticModuleMaker()
    {
        for ( ModuleMapIterator i = this->theModuleMap.begin();
                i != this->theModuleMap.end(); ++i )
        {
            delete i->second;
        }
    }

    virtual const Module& getModule( const std::string& aClassName, bool forceReload )
    {
        if ( this->theModuleMap.find( aClassName ) == this->theModuleMap.end() )
            throw DMException( "Can't find static module [" + aClassName + "]." );
        return  ( *this->theModuleMap[ aClassName ] );
    }

    /**
       Add a class to the subclass list.
       @param dm a pointer to a DynamicModule to be added
    */
    void addClass( Module* dm )
    {
        assert( dm );
        this->theModuleMap[ dm->getModuleName() ] = dm;
    }

    const ModuleMap& getModuleMap() const
    {
        return theModuleMap;
    }

protected:

    ModuleMap theModuleMap;

};

/**
  SharedModuleMaker dynamically instantiates various classes of
  objects encapsulated in shared object(.so) file.
  @sa StaticClassModuleMaker, SharedDynamicModule
*/

template<class T_>
class SharedModuleMaker: public StaticModuleMaker< T_ >
{
public:
    typedef StaticModuleMaker< T_ > Base;
    typedef SharedDynamicModule< T_ > SharedModule;

    SharedModuleMaker()
    {
        ; // do nothing
    }

    virtual ~SharedModuleMaker()
    {
        ; // do nothing
    }

    virtual const SharedModule& getModule( const std::string& aClassName, bool forceReload );

protected:
    void loadModule( const std::string& aClassname );
};


///////////////////////////// begin implementation
template< typename T_ > const SharedDynamicModule< T_ >&
SharedModuleMaker< T_ >::getModule(
        const std::string& aClassName, bool forceReload )
{

    if ( forceReload )
    {
        typename Base::ModuleMapIterator i ( this->theModuleMap.find( aClassName ) );
        if ( i != this->theModuleMap.end() )
        {
            delete i->second;
            this->theModuleMap.erase( i );
        }
    }

    if ( this->theModuleMap.find( aClassName ) == this->theModuleMap.end() )
    {
        loadModule( aClassName );
    }

    return *( ( SharedModule* ) this->theModuleMap[ aClassName ] );
}


template< typename T_ > void
SharedModuleMaker< T_ >::loadModule( const std::string& aClassname )
{
    // return immediately if already loaded
    if ( this->theModuleMap.find( aClassname ) != this->theModuleMap.end() )
    {
        return;
    }

    std::string filename( aClassname );
    lt_dlhandle handle( lt_dlopenext( filename.c_str() ) );
    if ( handle == NULL )
    {
        throw DMException( "Failed to find or load a DM [" + aClassname +
                           "]: " + lt_dlerror() );
    }
    addClass( new SharedModule( aClassname, filename, handle ) );
}

#endif /* __MODULEMAKER_HPP */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
