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


#ifndef __MODULEMAKER_HPP
#define __MODULEMAKER_HPP
#include <iostream>
#include <map>
#include <string>
#include <ltdl.h>
#include "DynamicModule.hpp"


/// doc needed

#define DynamicModuleEntry( T )\
addClass( new Module( std::string( #T ), &T::createInstance ) );

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
    if( aSearchPath == NULLPTR )
      {
	return "";
      }
    else
      {
	return aSearchPath;
      }
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
class StaticModuleMaker : public ModuleMaker
{

protected:

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

  virtual T* make( const std::string& classname );


  /**
     Add a class to the subclass list.
     @param dm a pointer to a DynamicModule to be added
  */

  void addClass( Module* dm );

protected:

  /*!
    This is only for internal use. Use public make() method.
 
    \param classname name of the class to be instantiated.
    \return pointer to a new instance.
  */

  virtual DMAllocator getAllocator( const std::string& classname ); 

protected:

  ModuleMap theModuleMap;

};

/**
  SharedModuleMaker dynamically instantiates various classes of 
  objects encapsulated in shared object(.so) file.
  @sa StaticClassModuleMaker, SharedDynamicModule
*/

template<class T,class DMAllocator=SimpleAllocator( T )>
class SharedModuleMaker : public StaticModuleMaker<T,DMAllocator>
{

public:

  typedef SharedDynamicModule<T> SharedModule;

  SharedModuleMaker();
  virtual ~SharedModuleMaker();

protected:


  virtual DMAllocator getAllocator( const std::string& classname );

  void loadModule( const std::string& classname );


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
  for( ModuleMapIterator i = theModuleMap.begin();
       i != theModuleMap.end(); ++i )
    {
      delete i->second;
    }
}

template<class T, class DMAllocator>
T* StaticModuleMaker<T,DMAllocator>::make( const std::string& classname ) 
{

  DMAllocator anAllocator( getAllocator( classname ) );
  if( anAllocator == NULL )
    {
      throw DMException( std::string( "unexpected error in " ) +
			 __PRETTY_FUNCTION__ );
    }

  T* anInstance( NULL );
  anInstance = anAllocator();

  if( anInstance == NULL )
    {
      throw DMException( "Can't instantiate [" + classname + "]." );
    }

  ++theNumberOfInstances;

  return anInstance;
}


template<class T,class DMAllocator>
void StaticModuleMaker<T,DMAllocator>::addClass( Module* dm )
{
  assert( dm );

  theModuleMap[ dm->getModuleName() ] = dm;
}


template<class T,class DMAllocator>
DMAllocator StaticModuleMaker<T,DMAllocator>::
getAllocator( const std::string& classname )
{
  if( theModuleMap.find( classname ) == theModuleMap.end() )
    {
      throw DMException( "Class [" + classname + "] not found." );
    }

  return theModuleMap[ classname ]->getAllocator();
}


////////////////////// SharedModuleMaker

template<class T,class DMAllocator>
SharedModuleMaker<T,DMAllocator>::SharedModuleMaker()
{
  int result = lt_dlinit();
  if( result != 0 )
    {
      std::cerr << "fatal: lt_dlinit() failed." << std::endl;
      exit( 1 );
    }
}

template<class T,class DMAllocator>
SharedModuleMaker<T,DMAllocator>::~SharedModuleMaker()
{
  int result = lt_dlexit();
  if( result != 0 )
    {
      std::cerr << "fatal: lt_dlexit() failed." << std::endl;
      exit( 1 );
    }
}

template<class T,class DMAllocator>
DMAllocator SharedModuleMaker<T,DMAllocator>::
getAllocator( const std::string& classname ) 
{
  DMAllocator anAllocator( NULL );

  try 
    {
      anAllocator = 
	StaticModuleMaker<T,DMAllocator>::getAllocator( classname );
    }
  catch( DMException& )
    { 
      // load module file and try again
      loadModule( classname );      
      anAllocator = 
	StaticModuleMaker<T,DMAllocator>::getAllocator( classname );
    }

  if( anAllocator == NULL )
    {
      // getAllocator() returned NULL! why?
      throw DMException( std::string("unexpected error in ") 
			 + __PRETTY_FUNCTION__ );
    }

  return anAllocator;
}

template<class T,class DMAllocator>
void SharedModuleMaker<T,DMAllocator>::loadModule( const std::string& classname )
{
  // return immediately if already loaded
  if( theModuleMap.find( classname ) != theModuleMap.end() )
    {
      return;      
    }
    
  SharedModule* sm( NULL );
  try 
    {
      sm = new SharedModule( classname );
      addClass( sm );
    }
  catch ( const DMException& e )
    {
      delete sm;
      
      throw;
    }
}

#endif /* __MODULEMAKER_HPP */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
