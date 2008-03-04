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
addClass( new Module( std::string( #T ), &T::createInstance, &T::getClassInfoPtr ) );

/**
   A base class for ModuleMakers
 */

class ModuleMaker
{

public:

  ModuleMaker() 
    : 
    theNumberOfInstances( 0 ) 
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
    if( error != 0 )
      {
	throw DMException( lt_dlerror() );
      }
  }

  static const std::string getSearchPath()
  {
    const char* aSearchPath( lt_dlgetsearchpath() );
    if( aSearchPath == 0 )
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

  /**
     @return the number of instance this have ever created
  */

  int getNumberOfInstances() const
  {
    return theNumberOfInstances;
  }

protected:

  int theNumberOfInstances;

};


/**
  StaticModuleMaker is used to instantiate
  various subclasses of certain template parameter class T. 
*/

template<class T, class DMAllocator = SimpleAllocator( T )>
class StaticModuleMaker
  : 
  public ModuleMaker
{

public:

  typedef DynamicModuleBase<T,DMAllocator> Module;
  typedef std::map<const std::string, Module*> ModuleMap;
  typedef typename ModuleMap::iterator ModuleMapIterator;

public: 

  StaticModuleMaker();
  virtual ~StaticModuleMaker();

  /**
     Instantiates given class of an object.
 
     @param classname name of class to be instantiated.
     @return pointer to a new instance.
  */

  virtual T* make( const std::string& aClassname );


  virtual const Module& getModule( const std::string& aClassName, bool forceReload )
  {
	if( this->theModuleMap.find( aClassName ) == this->theModuleMap.end() )
	  {
	      throw DMException( "Can't find static module [" + aClassName + "]." );

	  }
	
	return  (*this->theModuleMap[ aClassName ]);


  }


  /**
     Add a class to the subclass list.
     @param dm a pointer to a DynamicModule to be added
  */

  void addClass( Module* dm );

  const ModuleMap& getModuleMap() const
  {
    return theModuleMap;
  }

protected:

  /*!
    This is only for internal use. Use public make() method.
 
    \param classname name of the class to be instantiated.
    \return pointer to a new instance.
  */

  virtual DMAllocator getAllocator( const std::string& aClassname ); 


protected:

  ModuleMap theModuleMap;

};

/**
  SharedModuleMaker dynamically instantiates various classes of 
  objects encapsulated in shared object(.so) file.
  @sa StaticClassModuleMaker, SharedDynamicModule
*/

template<class T,class DMAllocator=SimpleAllocator( T )>
class SharedModuleMaker 
  : 
  public StaticModuleMaker<T,DMAllocator>
{

public:

  typedef SharedDynamicModule<T> SharedModule;

  SharedModuleMaker();
  virtual ~SharedModuleMaker();

  virtual const SharedModule& getModule( const std::string& aClassName, bool forceReload )
  {

    if ( forceReload ) 
      {
        typename StaticModuleMaker<T,DMAllocator>::ModuleMapIterator i ( this->theModuleMap.find( aClassName) );
        if( i != this->theModuleMap.end() )
          {
            delete i->second;
            this->theModuleMap.erase( i );
          }
    }
	if( this->theModuleMap.find( aClassName ) == this->theModuleMap.end() )
	  {
		loadModule( aClassName );
	  }
	return *((SharedModule*) this->theModuleMap[ aClassName ]);


  }


protected:


  virtual DMAllocator getAllocator( const std::string& aClassname );

  void loadModule( const std::string& aClassname );


};


///////////////////////////// begin implementation




////////////////////// StaticModuleMaker

template<class T,class DMAllocator>
StaticModuleMaker<T,DMAllocator>::StaticModuleMaker()
{
  ; // do nothing
}

template<class T,class DMAllocator>
StaticModuleMaker<T,DMAllocator>::~StaticModuleMaker()
{
  for( ModuleMapIterator i = this->theModuleMap.begin();
       i != this->theModuleMap.end(); ++i )
    {
      delete i->second;
    }
}

template<class T, class DMAllocator>
T* StaticModuleMaker<T,DMAllocator>::make( const std::string& aClassname ) 
{

  DMAllocator anAllocator( getAllocator( aClassname ) );
  if( !anAllocator )
    {
      throw DMException( std::string( "unexpected error in " ) +
			 __PRETTY_FUNCTION__ );
    }

  T* anInstance( 0 );
  anInstance = anAllocator();

  if( !anInstance )
    {
      throw DMException( "Can't instantiate [" + aClassname + "]." );
    }

  ++(this->theNumberOfInstances);

  return anInstance;
}


template<class T,class DMAllocator>
void StaticModuleMaker<T,DMAllocator>::addClass( Module* dm )
{
  assert( dm );

  this->theModuleMap[ dm->getModuleName() ] = dm;
}




template<class T,class DMAllocator>
DMAllocator StaticModuleMaker<T,DMAllocator>::
getAllocator( const std::string& aClassname )
{
  if( this->theModuleMap.find( aClassname ) == this->theModuleMap.end() )
    {
      throw DMException( "Class [" + aClassname + "] not found." );
    }

  return this->theModuleMap[ aClassname ]->getAllocator();
}


////////////////////// SharedModuleMaker

template<class T,class DMAllocator>
SharedModuleMaker<T,DMAllocator>::SharedModuleMaker()
{
}

template<class T,class DMAllocator>
SharedModuleMaker<T,DMAllocator>::~SharedModuleMaker()
{
}


template<class T,class DMAllocator>
DMAllocator SharedModuleMaker<T,DMAllocator>::
getAllocator( const std::string& aClassname ) 
{
  DMAllocator anAllocator( 0 );

  try 
    {
      anAllocator = 
	StaticModuleMaker<T,DMAllocator>::getAllocator( aClassname );
    }
  catch( DMException& )
    { 
      // load module file and try again
      loadModule( aClassname );      
      anAllocator = 
	StaticModuleMaker<T,DMAllocator>::getAllocator( aClassname );
    }

  if( !anAllocator )
    {
      // getAllocator() returned NULL! why?
      throw DMException( std::string( "unexpected error in " ) 
			 + __PRETTY_FUNCTION__ );
    }

  return anAllocator;
}

template<class T,class DMAllocator>
void 
SharedModuleMaker<T,DMAllocator>::loadModule( const std::string& aClassname )
{
  // return immediately if already loaded
  if( this->theModuleMap.find( aClassname ) != this->theModuleMap.end() )
    {
      return;      
    }
    
  SharedModule* aSharedModule( 0 );
  std::string filename( aClassname );
  lt_dlhandle handle( lt_dlopenext( filename.c_str() ) );
  if ( !handle ) 
    {
      throw DMException( "Failed to find or load a DM [" + aClassname + 
			 "]: " + lt_dlerror() );
    }
  typename SharedModule::DMAllocator anAllocator(
      *reinterpret_cast< DMAllocator* >(
	lt_dlsym( handle, "CreateObject" ) ) );
  if ( !anAllocator )
    {
      throw DMException( "[" + filename + "] is not a valid DM file: "
			  + lt_dlerror() );  
    }
  InfoLoaderType anInfoLoader(
      *reinterpret_cast< InfoLoaderType* >(
	lt_dlsym( handle, "GetClassInfo" ) ) );
  if ( !anInfoLoader )
    {
      throw DMException( "[" + filename + "] is not a valid DM file: "
			  + lt_dlerror() );  
    }

  const char* typeString = *reinterpret_cast< const char ** >(
      lt_dlsym( handle, "__DM_TYPE" ) );
  if ( !typeString )
    {
      throw DMException( "[" + filename + "] is not a valid DM file: "
			  + lt_dlerror() );  
    }
  addClass( new SharedModule( aClassname, anAllocator, anInfoLoader,
                              typeString, filename, handle ) );
}

#endif /* __MODULEMAKER_HPP */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
