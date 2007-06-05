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
//
// authors:
// Koichi Takahashi <shafi@e-cell.org>
// Nayuta Iwata
//
// E-Cell Project.
//

#ifndef __PYTHONPROCESSBASE_HPP
#define __PYTHONPROCESSBASE_HPP

#include <Python.h>
#include <object.h>
#include <compile.h>
#include <eval.h>
#include <frameobject.h>

#include "boost/python.hpp"

#include "libecs.hpp"

#include "FullID.hpp"

#include "Process.hpp"


USE_LIBECS;

namespace python = boost::python;

LIBECS_DM_CLASS( PythonProcessBase, Process )
{

DECLARE_ASSOCVECTOR
  ( String, Polymorph, std::less<const String>, PropertyMap );

public:

  LIBECS_DM_OBJECT_ABSTRACT( PythonProcessBase )
    {
      INHERIT_PROPERTIES( Process );
    }


  PythonProcessBase()
  {
    if( ! Py_IsInitialized() )
      {
	THROW_EXCEPTION( UnexpectedError, getClassNameString() + 
			 ": Python interpreter is not initialized." );
      }

    python::handle<> aHandle( python::borrowed( PyImport_GetModuleDict() ) );
    python::dict aModuleList( aHandle );

    if( ! aModuleList.has_key( python::str( "ecell.ecs" ) ) )
      {
	THROW_EXCEPTION( UnexpectedError, getClassNameString() + 
			 ": ecell.ecs module must be imported before" +
			 " using this class." );
      }

  }

  virtual ~PythonProcessBase()
  {
    // ; do nothing
  }


  python::object compilePythonCode( StringCref aPythonCode, 
				    StringCref aFilename,
				    int start )
  {
    return python::object( python::handle<>
			   ( Py_CompileString( const_cast<char*>
					       ( aPythonCode.c_str() ),
					       const_cast<char*>
					       ( aFilename.c_str() ),
					       start ) ) );
  }

  void defaultSetProperty( StringCref aPropertyName,
			   PolymorphCref aValue )
    {
      theLocalNamespace[ aPropertyName ] =
	python::object( python::handle<>( PyFloat_FromDouble( aValue ) ) );

      thePropertyMap[ aPropertyName ] = aValue;
    }

  const Polymorph defaultGetProperty( StringCref aPropertyName ) const
    {
      PropertyMapConstIterator
	aPropertyMapIterator( thePropertyMap.find( aPropertyName ) );

      if( aPropertyMapIterator != thePropertyMap.end() )
	{
	  return aPropertyMapIterator->second;
	}
      else
	{
	  THROW_EXCEPTION( NoSlot, getClassNameString() + " : Property [" +
			   aPropertyName + "] is not defined" );
	}
    }

  const Polymorph defaultGetPropertyList() const
    {
      PolymorphVector aVector;

      for( PropertyMapConstIterator 
	     aPropertyMapIterator( thePropertyMap.begin() );
	   aPropertyMapIterator != thePropertyMap.end();
	   ++aPropertyMapIterator )
	{
	  aVector.push_back( aPropertyMapIterator->first );
	}      

      return aVector;
    }

  const Polymorph
    defaultGetPropertyAttributes( StringCref aPropertyName ) const
    {
      PolymorphVector aVector;

      Integer aPropertyFlag( 1 );

      aVector.push_back( aPropertyFlag ); //isSetable
      aVector.push_back( aPropertyFlag ); //isGetable
      aVector.push_back( aPropertyFlag ); //isLoadable
      aVector.push_back( aPropertyFlag ); //isSavable

      return Polymorph( aVector );
    }

  virtual void initialize();

protected:

  python::dict   theGlobalNamespace;
  python::dict   theLocalNamespace;

  PropertyMap    thePropertyMap;
};


void PythonProcessBase::initialize()
{
  Process::initialize();
  
  theGlobalNamespace.clear();

  for( VariableReferenceVectorConstIterator 
	 i( getVariableReferenceVector().begin() );
       i != getVariableReferenceVector().end(); ++i )
    {
      VariableReferenceCref aVariableReference( *i );

      theGlobalNamespace[ aVariableReference.getName() ] = 
      	python::object( boost::ref( aVariableReference ) );
    }

  // extract 'this' Process's methods and attributes
  python::object 
    aPySelfProcess( python::ptr( static_cast<Process*>( this ) ) );
  //  python::dict aSelfDict( aPySelfProcess.attr("__dict__") );

  theGlobalNamespace[ "self" ] = aPySelfProcess;
  //  theGlobalNamespace.update( aSelfDict );

  python::handle<> 
    aMainModule( python::borrowed( PyImport_AddModule( "__main__" ) ) );
  python::handle<> 
    aMathModule( python::borrowed( PyImport_AddModule( "math" ) ) );

  python::handle<> 
    aMainNamespace( python::borrowed
		    ( PyModule_GetDict( aMainModule.get() ) ) );
  python::handle<> 
    aMathNamespace( python::borrowed
  		    ( PyModule_GetDict( aMathModule.get() ) ) );

  theGlobalNamespace.update( aMainNamespace );
  theGlobalNamespace.update( aMathNamespace );

}

LIBECS_DM_INIT_STATIC( PythonProcessBase, Process );

#endif /* __PYTHONPROCESSBASE_HPP */
