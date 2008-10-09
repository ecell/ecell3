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

#include <exception>
#include <string>
#include "ltdl.h"

#if defined(WIN32) || defined(_WIN32)
#undef GetClassInfo
#endif /* _WIN32 */

/// doc needed

#define SimpleAllocator( BASE ) BASE* (*)()
typedef const void*(*InfoLoaderType)();

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
  
template <class Base,class _TAllocator = SimpleAllocator( Base )>
class DynamicModuleBase
{
public:

  typedef _TAllocator DMAllocator;

public:		

  DynamicModuleBase( const std::string& moduleName,
                     DMAllocator allocator,
                     InfoLoaderType infoLoader,
                     const std::string& typeName = "" );
  virtual ~DynamicModuleBase(){}
  
  const std::string& getModuleName() const
  {
    return this->theModuleName;
  }

  virtual const std::string getFileName() const
  {
    return "";
  }

  const DMAllocator& getAllocator() const
  {
    return this->theAllocator;
  }

  const InfoLoaderType& getInfoLoader() const
  {
	return this->theInfoLoader;
  }

  const std::string getTypeName() const
  {
    return this->theTypeName;
  }

protected:

  const std::string theModuleName;
  DMAllocator theAllocator;
  InfoLoaderType theInfoLoader;
  const std::string theTypeName;
};


/**
   DynamicModule instantiates objects of a single class.
*/

template <class Base,class Derived,class DMAllocator = SimpleAllocator( Base )>
class DynamicModule
  :
  public DynamicModuleBase< Base, DMAllocator >
{

public:

  DynamicModule( const std::string& moduleName,
		 const std::string& typeName = "" );
  virtual ~DynamicModule(){}
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

template <class Base,class DMAllocator = SimpleAllocator( Base )>
class SharedDynamicModule : public DynamicModuleBase< Base >
{

public:

  SharedDynamicModule( const std::string& classname, DMAllocator allocator,
                       InfoLoaderType infoLoader, const std::string& typeName,
                       const std::string& fileName, lt_dlhandle handle );
  virtual ~SharedDynamicModule();
  const std::string getFileName() const;

private:

  lt_dlhandle theHandle;
  std::string theFileName;
  std::string theTypeName;
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

template < class Base, class DMAllocator >
DynamicModuleBase< Base, DMAllocator >::
DynamicModuleBase( const std::string& moduleName,
		   DMAllocator allocator, InfoLoaderType infoLoader,
		   const std::string& typeName )
  : 
  theModuleName( moduleName ),
  theAllocator( allocator ),
  theInfoLoader( infoLoader ),
  theTypeName( typeName )
{
  ; // do nothing
}

template < class Base, class Derived, class DMAllocator >
DynamicModule< Base, Derived, DMAllocator >::
DynamicModule( const std::string& moduleName, const std::string& typeName )
  : 
  DynamicModuleBase<Base,DMAllocator>( moduleName,
				       reinterpret_cast< DMAllocator >(
                         &Derived::createInstance ),
				       &Derived::getClassInfoPtr,
				       typeName )
{
  ; // do nothing
}

template < class Base, class DMAllocator >
SharedDynamicModule< Base, DMAllocator >::
SharedDynamicModule( const std::string& classname, DMAllocator allocator,
                     InfoLoaderType infoLoader, const std::string& typeName,
                     const std::string& fileName, lt_dlhandle handle )
  :
  DynamicModuleBase<Base,DMAllocator>( classname, allocator, infoLoader,
                                       typeName ), 
  theFileName( fileName ),
  theHandle( handle )
{
}

template < class Base, class DMAllocator >
SharedDynamicModule<Base,DMAllocator>::~SharedDynamicModule()
{

  if( this->theHandle )
    {
      lt_dlclose( this->theHandle );
      this->theHandle = 0;
    }
}



template < class Base, class DMAllocator >
const std::string SharedDynamicModule<Base,DMAllocator>::getFileName() const
{
  return this->theFileName;
}

#endif /* __DYNAMICMODULE_HPP */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/

