//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
// 		This file is part of dmtool package
//
//	       written by Kouichi Takahashi  <shafi@sfc.keio.ac.jp>
//
//                              E-CELL Project,
//                          Lab. for Bioinformatics,  
//                             Keio University.
//
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// dmtool is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// dmtool is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with dmtool -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER


#ifndef __DYNAMICMODULE_HPP
#define __DYNAMICMODULE_HPP

#include <exception>

#include <string>
#include <ltdl.h>


/// doc needed

#define SimpleAllocator( BASE ) BASE* (*)()


/**
   Exception class for dmtool.
*/

class DMException : public std::exception
{
public: 

  DMException( const std::string& message ) : theMessage( message )
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
  
template <class Base,class DMAllocator = SimpleAllocator( Base )>
class DynamicModuleBase
{

public:		

  DynamicModuleBase( const std::string& Modulename, DMAllocator Allocator );

  const std::string& getModuleName() const
  {
    return theModuleName;
  }

  const std::string& getFileName() const
  {
    return getModuleName();
  }

  const DMAllocator& getAllocator() const
  {
    return theAllocator;
  }

protected:

  std::string theModuleName;
  DMAllocator theAllocator;

};


/**
   DynamicModule instantiates objects of a single class.
*/

template <class Base,class Derived,class DMAllocator = SimpleAllocator( Base )>
class DynamicModule
  :
  public DynamicModuleBase< Base,DMAllocator >
{

public:

  DynamicModule( const std::string& modulename );

};



/**
   SharedDynamicModule loads a class from a shared object file
   and instantiate objects of the loaded class.
   It opens and loads a shared object(.so) file into memory
   when constructed. It closes the file when deleted.

   The shared object must have followings:
     - T* CreateObject()       - which returns a new object.
   and
     - a full set of symbols needed to instantiate and use the class.
*/

template <class Base,class DMAllocator = SimpleAllocator( Base )>
class SharedDynamicModule : public DynamicModuleBase< Base >
{

public:

  SharedDynamicModule( const std::string& classname, 
		       const std::string& directory );
  virtual ~SharedDynamicModule();

  const std::string& getFileName() const;

private:

  lt_dlhandle theHandle;

};

/**
   comments needed
*/

#define NewDynamicModule( BASE, DERIVED )\
addClass( new DynamicModule< BASE, DERIVED >( #DERIVED ) )

/**
   comments needed
*/

#define NewDynamicModuleWithAllocator( BASE, DERIVED, ALLOC )\
addClass( new DynamicModule< BASE, DERIVED, ALLOC >( #DERIVED ) )

/// an allocator function template

template< class Base, class Derived >
Base* CreateObject()
{
  return new Derived;
}


//////////////////////////// begin implementation

template < class Base, class DMAllocator >
DynamicModuleBase< Base, DMAllocator >::
DynamicModuleBase( const std::string& modulename,
		   DMAllocator allocator )
  : 
  theModuleName( modulename ),
  theAllocator( allocator )
{
  ; // do nothing
}

template < class Base, class Derived, class DMAllocator >
DynamicModule< Base, Derived, DMAllocator >::
DynamicModule( const std::string& modulename )
  : 
  DynamicModuleBase<Base,DMAllocator>( modulename, &Derived::instance )
{
  ; // do nothing
}

template < class Base, class DMAllocator >
SharedDynamicModule< Base, DMAllocator >::
SharedDynamicModule( const std::string& classname,
		     const std::string& directory ) 
  :
  DynamicModuleBase<Base,DMAllocator>( classname, NULL ), 
  theHandle( NULL )
{
  std::string filename( directory + '/' + classname );
  theHandle = lt_dlopenext( filename.c_str() );

  if( theHandle == NULL ) 
    {
      throw DMException( lt_dlerror() );
    }

  theAllocator = *((DMAllocator*)( lt_dlsym( theHandle, "CreateObject" ) ));

  if( theAllocator == NULL )
    {
      throw DMException( lt_dlerror() );  
    }
}

template < class Base, class DMAllocator >
SharedDynamicModule<Base,DMAllocator>::~SharedDynamicModule()
{
  if( theHandle != NULL )
    {
      lt_dlclose( theHandle );
    }
}

template < class Base, class DMAllocator >
const std::string& SharedDynamicModule<Base,DMAllocator>::getFileName() const
{
  return string( theHandle->filename );
}

#endif /* __DYNAMICMODULE_HPP */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
