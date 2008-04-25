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

#ifndef __DYNAMICMODULE_HPP
#define __DYNAMICMODULE_HPP

#include <exception>
#include <string>
#include <map>
#include <vector>
#include "ltdl.h"

#if defined(WIN32) || defined(_WIN32)
#undef GetClassInfo
#endif /* _WIN32 */

/// doc needed
/**
   Exception class for dmtool.
*/

class DMException : public std::exception
{
public:

    DMException( const std::string& message )
            :
            theMessage( message )
    {
        ; // do nothing
    }

    ~DMException() throw()
    {
        ; // do nothing
    }

    /**
       Get dynamically created exception message.
     */

    const char* what() const throw()
    {
        return theMessage.c_str();
    }

private:

    const std::string theMessage;

};

/**
  Common base class of DynamicModule and SharedDynamicModule
*/

template <typename T_>
class DynamicModuleBase
{
public:
    typedef DynamicModuleBase Module;

public:

    DynamicModuleBase( const std::string& moduleName )
        : theModuleName( moduleName )
    {
        ; // do nothing
    }

    virtual ~DynamicModuleBase()
    {
        ; // do nothing
    }

    const std::string& getModuleName() const
    {
        return theModuleName;
    }

    virtual const std::string& getFileName() const = 0;

    virtual T_* createInstance() const = 0;

    virtual const void* getInfo( const std::string& kind ) const = 0;

protected:
    const std::string theModuleName;
};


/**
   DynamicModule instantiates objects of a single class.
*/

template <typename T_>
class DynamicModule: public DynamicModuleBase< T_ >
{
public:
    typedef DynamicModuleBase< T_ > Module;

public:
    DynamicModule( const std::string& moduleName )
        : DynamicModuleBase<T_>( moduleName )
    {
        ; // do nothing
    }

    virtual ~DynamicModule()
    {
        // do nothing
    }

    virtual T_* createInstance() const
    {
        return new T_( this );
    }

    virtual const void* getInfo( const std::string& kind ) const
    {
        return 0;
    }

    virtual const std::string& getFileName() const
    {
        static const std::string empty("");
        return empty;
    }
};



/**
   SharedDynamicModule loads a class from a shared object file
   and instantiate objects of the loaded class.
   It opens and loads a shared object(.so) file into memory
   when constructed. It closes the file when deleted.

   The shared object must have followings:
     - T* CreateObject()       - which returns a new object.
   and
     - void* GetClassInfo()    - which must be reinterpreted to PolymorphMap in libecs, sorry,
  maybe later a pure string version should be implemented
     - a full set of symbols needed to instantiate and use the class.
*/

template <class T_>
class SharedDynamicModule : public DynamicModuleBase< T_ >
{
public:
    typedef DynamicModuleBase< T_ > Module;

protected:
    typedef T_* (*FactoryFunction)( const Module* );
    typedef const void* (*GetInfoFunction)( const std::string& kind );

public:

    SharedDynamicModule( const std::string& classname,
                         const std::string& fileName, lt_dlhandle handle );

    virtual ~SharedDynamicModule()
    {
        if ( theHandle )
        {
            lt_dlclose( theHandle );
            theHandle = 0;
        }
    }

    virtual T_* createInstance() const
    {
        return theFactoryFunction( this );
    }

    virtual const void* getInfo( const std::string& kind ) const
    {
        return theInfoRetriever( kind );
    }

    virtual const std::string& getFileName() const
    {
        return theFileName;
    }

private:
    lt_dlhandle theHandle;
    std::string theFileName;
    FactoryFunction theFactoryFunction;
    GetInfoFunction theInfoRetriever;
};

/**
   comments needed
*/

#define NewDynamicModule( BASE, DERIVED )\
addClass( new DynamicModule< BASE, DERIVED >( #DERIVED, DERIVED::getTypeName() ) )

/**
   comments needed
*/

#define NewDynamicModuleWithAllocator( BASE, DERIVED, ALLOC )\
addClass( new DynamicModule< BASE, DERIVED, ALLOC >( #DERIVED ) )


//////////////////////////// begin implementation

template < typename TBase_ >
SharedDynamicModule< TBase_ >::SharedDynamicModule(
    const std::string& classname, const std::string& fileName,
    lt_dlhandle handle )
    : DynamicModuleBase< TBase_ >( classname ),
      theFileName( fileName ),
      theHandle( handle ),
      theFactoryFunction( 0 ),
      theInfoRetriever( 0 )
{
    FactoryFunction* anFactoryFunctionPtr( 0 );
    GetInfoFunction* anGetInfoFunctionPtr( 0 );
    const char** aTypeStringPtr( 0 );

    anFactoryFunctionPtr = reinterpret_cast< FactoryFunction* >(
                         lt_dlsym( handle, "CreateObject" ) );
    if ( !anFactoryFunctionPtr )
        goto fail_dlsym;
    anGetInfoFunctionPtr = reinterpret_cast< GetInfoFunction* >(
                              lt_dlsym( handle, "GetClassInfo" ) );
    if ( !anGetInfoFunctionPtr )
        goto fail_dlsym;

    if ( !*anFactoryFunctionPtr
            || !*anGetInfoFunctionPtr
            || !*aTypeStringPtr )
    {
        throw DMException( "[" + fileName + "] is not a valid DM file." );
    }

    theFactoryFunction = *anFactoryFunctionPtr;
    theInfoRetriever = *anGetInfoFunctionPtr;
    return;

fail_dlsym:
    throw DMException( "[" + fileName + "] is not a valid DM file: "
                       + lt_dlerror() );
}

#endif /* __DYNAMICMODULE_HPP */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
