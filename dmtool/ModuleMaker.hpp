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
//#include <ltdl.h>
#include <dlfcn.h>
#include "DynamicModule.hpp"


using namespace std;

/// doc needed

#define DynamicModuleEntry( T )\
addClass( new Module( string( #T ), &T::createInstance ) );

/**
   A base class for ModuleMakers
 */

template<class T, class DMAllocator = SimpleAllocator(T)>
class ModuleMakerBase
{

public:

  ModuleMakerBase();
  virtual ~ModuleMakerBase();

  /**
     Instantiates given class of an object.
 
     @param classname name of class to be instantiated.
     @return pointer to a new instance.
  */

  virtual T* make( const string& classname );

  /**
     @return the number of instance this have ever created
  */

  int getNumberOfInstances() const
  {
    return theNumInstance;
  }

protected:

  int theNumberOfInstances;

  virtual DMAllocator getAllocator( const string& classname ) = 0;
};


/**
  StaticModuleMaker is used to instantiate
  various subclasses of certain template parameter class T. 
*/

template<class T, class DMAllocator = SimpleAllocator( T )>
class StaticModuleMaker : public ModuleMakerBase<T,DMAllocator>
{

protected:

  typedef DynamicModuleBase<T,DMAllocator> Module;
  typedef map<const string, Module*> ModuleMap;
  typedef typename ModuleMap::iterator ModuleMapIterator;

public: 

  StaticModuleMaker();
  virtual ~StaticModuleMaker();

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

  virtual DMAllocator getAllocator( const string& classname ); 

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

  void setSearchPath( const string& path );

  const string& getSearchPath() const;

protected:

  string theSearchPathString;

  virtual DMAllocator getAllocator( const string& classname );

  void loadModule( const string& classname );


};


///////////////////////////// begin implementation


////////////////////// ModuleMakerBase

template<class T, class DMAllocator>
ModuleMakerBase<T,DMAllocator>::ModuleMakerBase() 
  : 
  theNumberOfInstances( 0 ) 
{
  ; // do nothing
}

template<class T, class DMAllocator>
ModuleMakerBase<T,DMAllocator>::~ModuleMakerBase() 
{
  ; // do nothing
}

template<class T, class DMAllocator>
T* ModuleMakerBase<T,DMAllocator>::make( const string& classname ) 
{

  DMAllocator anAllocator( getAllocator( classname ) );
  if( anAllocator == NULL )
    {
      throw DMException( string( "unexpected error in " ) +
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

template<class T,class DMAllocator>
void StaticModuleMaker<T,DMAllocator>::addClass( Module* dm )
{
  assert( dm );

  theModuleMap[ dm->getModuleName() ] = dm;
}


template<class T,class DMAllocator>
DMAllocator StaticModuleMaker<T,DMAllocator>::
getAllocator( const string& classname )
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
  :
  theSearchPathString( "." )
{
  /*
    int result = lt_dlinit();
    if( result != 0 )
    {
    cerr << "warning: lt_dlinit() failed." << endl;
    }
  */
}

template<class T,class DMAllocator>
SharedModuleMaker<T,DMAllocator>::~SharedModuleMaker()
{
  /*
    int result = lt_dlexit();
    if( result != 0 )
    {
    cerr << "warning: lt_dlexit() failed." << endl;
    }
  */
}

template<class T,class DMAllocator>
DMAllocator SharedModuleMaker<T,DMAllocator>::
getAllocator( const string& classname ) 
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
      throw DMException( string("unexpected error in ") 
			 + __PRETTY_FUNCTION__ );
    }

  return anAllocator;
}

template<class T,class DMAllocator>
void SharedModuleMaker<T,DMAllocator>::loadModule( const string& classname )
{
  // return immediately if already loaded
  if( theModuleMap.find( classname ) != theModuleMap.end() )
    {
      return;      
    }
    
  // iterate over the search path
  string::size_type aTail( 0 );
  do
    {
      string::size_type aHead( aTail );
      aTail = theSearchPathString.find_first_of( ":", aHead );
      string aDirectory = theSearchPathString.substr( aHead, aTail - aHead );

      if( aDirectory == "" )
	{
	  continue;
	}

      SharedModule* sm( NULL );
      try 
	{
	  sm = new SharedModule( classname, aDirectory );
	  addClass( sm );
	  break;
	}
      catch ( const DMException& e )
	{
	  delete sm;

	  // re-throw if failed for the last item
	  if( aTail == string::npos )
	    {
	      throw DMException( "failed to load [" + classname + 
				 "]: " + e.what() );
	    }
	}

    } while( aTail != string::npos );

}

template<class T,class DMAllocator>
void SharedModuleMaker<T,DMAllocator>::setSearchPath( const string& path )
{
  theSearchPathString = path;
}

template<class T,class DMAllocator>
const string& SharedModuleMaker<T,DMAllocator>::getSearchPath() const
{
  return theSearchPathString;
}

#endif /* __MODULEMAKER_HPP */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
