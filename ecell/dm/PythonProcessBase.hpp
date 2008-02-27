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
#include <boost/python.hpp>

#include "libecs/libecs.hpp"
#include "libecs/FullID.hpp"
#include "libecs/Process.hpp"

LIBECS_DM_CLASS( PythonProcessBase, libecs::Process )
{

    DECLARE_ASSOCVECTOR(
        libecs::String,
        libecs::Polymorph,
        std::less<const libecs::String>,
        PropertyMap );

public:

    LIBECS_DM_OBJECT_ABSTRACT( PythonProcessBase )
    {
        INHERIT_PROPERTIES( Process );
    }


    PythonProcessBase()
    {
        if ( ! Py_IsInitialized() ) {
            Py_Initialize();
        }
    }

    virtual ~PythonProcessBase()
    {
        // ; do nothing
    }


    boost::python::object compilePythonCode( libecs::StringCref aPythonCode,
                                           libecs::StringCref aFilename,
                                           int start )
    {
        return boost::python::object( boost::python::handle<>
                                    ( Py_CompileString( const_cast<char*>
                                                        ( aPythonCode.c_str() ),
                                                        const_cast<char*>
                                                        ( aFilename.c_str() ),
                                                        start ) ) );
    }

    void defaultSetProperty( libecs::StringCref aPropertyName,
                             libecs::PolymorphCref aValue )
    {
        theLocalNamespace[ aPropertyName ] =
            boost::python::object( boost::python::handle<>( PyFloat_FromDouble( aValue ) ) );

        thePropertyMap[ aPropertyName ] = aValue;
    }

    const libecs::Polymorph defaultGetProperty( libecs::StringCref aPropertyName ) const
    {
        PropertyMapConstIterator
        aPropertyMapIterator( thePropertyMap.find( aPropertyName ) );

        if ( aPropertyMapIterator != thePropertyMap.end() ) {
            return aPropertyMapIterator->second;
        } else {
            THROW_EXCEPTION( libecs::NoSlot,
                             getClassNameString() + " : Property [" +
                             aPropertyName + "] is not defined" );
        }
    }

    const libecs::Polymorph defaultGetPropertyList() const
    {
        libecs::PolymorphVector aVector;

        for ( PropertyMapConstIterator
                aPropertyMapIterator( thePropertyMap.begin() );
                aPropertyMapIterator != thePropertyMap.end();
                ++aPropertyMapIterator ) {
            aVector.push_back( aPropertyMapIterator->first );
        }

        return aVector;
    }

    const libecs::Polymorph
    defaultGetPropertyAttributes( libecs::StringCref aPropertyName ) const
    {
        libecs::PolymorphVector aVector;

        libecs::Integer aPropertyFlag( 1 );

        aVector.push_back( aPropertyFlag ); //isSetable
        aVector.push_back( aPropertyFlag ); //isGetable
        aVector.push_back( aPropertyFlag ); //isLoadable
        aVector.push_back( aPropertyFlag ); //isSavable

        return libecs::Polymorph( aVector );
    }

    virtual void initialize();

protected:

    boost::python::dict   theGlobalNamespace;
    boost::python::dict   theLocalNamespace;

    PropertyMap    thePropertyMap;
};


void PythonProcessBase::initialize()
{
    Process::initialize();

    theGlobalNamespace.clear();

    for ( libecs::VariableReferenceVectorConstIterator
            i( getVariableReferenceVector().begin() );
            i != getVariableReferenceVector().end(); ++i ) {
        libecs::VariableReferenceCref aVariableReference( *i );

        theGlobalNamespace[ aVariableReference.getName() ] =
            boost::python::object( boost::ref( aVariableReference ) );
    }

    // extract 'this' Process's methods and attributes
    boost::python::object
    aPySelfProcess( boost::python::ptr( static_cast<Process*>( this ) ) );
    //  boost::python::dict aSelfDict( aPySelfProcess.attr("__dict__") );

    theGlobalNamespace[ "self" ] = aPySelfProcess;
    //  theGlobalNamespace.update( aSelfDict );

    boost::python::handle<>
    aMainModule( boost::python::borrowed( PyImport_AddModule( "__main__" ) ) );
    boost::python::handle<>
    aMathModule( boost::python::borrowed( PyImport_AddModule( "math" ) ) );

    boost::python::handle<>
    aMainNamespace( boost::python::borrowed
                    ( PyModule_GetDict( aMainModule.get() ) ) );
    boost::python::handle<>
    aMathNamespace( boost::python::borrowed
                    ( PyModule_GetDict( aMathModule.get() ) ) );

    theGlobalNamespace.update( aMainNamespace );
    theGlobalNamespace.update( aMathNamespace );
}

LIBECS_DM_INIT_STATIC( PythonProcessBase, libecs::Process );

#endif /* __PYTHONPROCESSBASE_HPP */
