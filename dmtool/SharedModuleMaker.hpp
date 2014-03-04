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


#ifndef __SHAREDMODULEMAKER_HPP
#define __SHAREDMODULEMAKER_HPP

#include "ModuleMaker.hpp"
#include "SharedDynamicModule.hpp"
#include "SharedModuleMakerInterface.hpp"
#include <fstream>

class SharedModuleMakerBase: public SharedModuleMakerInterface
{
public:
    virtual ~SharedModuleMakerBase() {}

    virtual void setSearchPath( const std::string& path )
    {
        theSearchPath.clear();
        for ( std::string::size_type i( 0 ), end( path.size() ), next( 0 );
              i < end; i = next + 1 )
        {
            next = path.find( PATH_SEPARATOR, i );
            next = next == std::string::npos ? end: next;
            if ( next > i )
            {
                theSearchPath.insert( path.substr( i, next - i ) );
            }
        }
    }

    virtual std::string getSearchPath() const
    {
        typedef std::set< std::string > StringSet;
        std::string retval;

        for ( StringSet::const_iterator i( theSearchPath.begin() );
              i != theSearchPath.end(); ++i )
        {
            retval += (*i);
            retval += PATH_SEPARATOR; 
        }
        if ( !retval.empty() )
            retval.resize( retval.size() - 1 );
        return retval;
    }
    /**
        Initializes the dynamic module facility.
        Applications that use this library must call this function
        prior to any operation involved with the facility.
        @return true on error, false otherwise.
     */
    static bool initialize()
    {
        if ( lt_dlinit() > 0 )
            return true;

        if ( lt_dlsetsearchpath("") > 0 )
        {
            lt_dlexit();
            return true;
        }

        return false;
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

protected:
    std::set< std::string > theSearchPath;
};

/**
    SharedModuleMaker dynamically instantiates various classes of 
    objects encapsulated in shared object(.so) file.
    @sa StaticClassModuleMaker, SharedDynamicModule
*/

template<class T >
class SharedModuleMaker : public ModuleMaker< T >,
                          public SharedModuleMakerBase
{
public:
    typedef ModuleMaker< T > Base;
    typedef DynamicModule< T > Module;
    typedef SharedDynamicModule< T > SharedModule;

    SharedModuleMaker()
    {
        ; // do nothing
    }

    virtual ~SharedModuleMaker()
    {
        ; // do nothing
    }

    virtual const Module& getModule( const std::string& aClassName, bool forceReload = false )
    {
        if ( forceReload ) 
        {
            typename Base::ModuleMap::iterator i( this->theModuleMap.find( aClassName ) );
            if ( i != this->theModuleMap.end() &&
                 (*i).second->getType() == DM_TYPE_SHARED )
            {
                this->theModuleMap.erase( i );
                delete i->second;
            }
        }

        if ( this->theModuleMap.find( aClassName ) == this->theModuleMap.end() )
        {
            loadModule( aClassName );
        }

        return *this->theModuleMap[ aClassName ];
    }


protected:
    void loadModule( const std::string& aClassname )
    {
        typedef std::set< std::string > StringSet;

        std::string filename;

        lt_dlhandle handle( 0 );
        {
            std::string error;

            for ( StringSet::const_iterator i( this->theSearchPath.begin() );
                  i != this->theSearchPath.end(); ++i )
            {
#ifdef LTDL_SHLIB_EXT
                filename = (*i) + '/' + aClassname + LTDL_SHLIB_EXT;
                if ( !std::ifstream( filename.c_str() ).is_open() )
                {
                    continue;
                }

                handle = lt_dlopen( filename.c_str() );
#else
                filename = (*i) + '/' + aClassname;
                handle = lt_dlopenext( filename.c_str() );
#endif /* LTDL_SHLIB_EXT */
                if ( handle ) 
                {
                    break;
                }
                const char* err( lt_dlerror() );
                if ( error.empty() )
                {
                    error = err ? err: "unknown reasons";
                }
                if ( !err )
                {
                    error = "various reasons";
                }
                  
            }

            if ( !handle )
            {
                if ( error.empty() )
                {
                    throw DMException( "Failed to find DM ["
                                       + aClassname + "]" );
                }
                else
                {
                    throw DMException( "Failed to load DM ["
                                       + aClassname + "]: "
                                       + error );
                }
            }
        }

        DynamicModuleDescriptor* desc(
                reinterpret_cast< DynamicModuleDescriptor* >(
                    lt_dlsym( handle, "__dm_descriptor" ) ) );
        if ( !desc )
        {
            throw DMException( "[" + filename + "] is not a valid DM file." );
        }

        if ( aClassname != desc->moduleName )
        {
            throw DMException( "[" + filename + "] is compiled as the module [" + desc->moduleName + "]." );
        }

        this->addClass( new SharedModule( *desc, filename, handle ) );
    }
};

#endif /* __SHAREDMODULEMAKER_HPP */
