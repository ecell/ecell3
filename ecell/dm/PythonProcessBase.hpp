//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 2003 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-CELL is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-CELL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-CELL -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// authors:
// Kouichi Takahashi <shafi@e-cell.org>
// Nayuta Iwata
//
// E-CELL Project, Lab. for Bioinformatics, Keio University.
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
#include "Variable.hpp"
#include "PropertyInterface.hpp"

#include "FullID.hpp"

#include "Process.hpp"


USE_LIBECS;

namespace python = boost::python;

//LIBECS_DM_CLASS( PythonProcessBase, Process )
class PythonProcessBase
  :
  public Process
{

public:

  //  LIBECS_DM_OBJECT_ABSTRACT( PythonProcessBase )
  //    {
  //      INHERIT_PROPERTIES( Process );
  //    }


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

  virtual void initialize();

protected:

  python::dict   theGlobalNamespace;
  python::dict   theLocalNamespace;

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
    aMainNamespace( python::borrowed
		    ( PyModule_GetDict( aMainModule.get() ) ) );

  theGlobalNamespace.update( aMainNamespace );

}

#endif /* __PYTHONPROCESSBASE_HPP */
