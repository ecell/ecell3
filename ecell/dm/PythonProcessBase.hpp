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

#include <libecs/libecs.hpp>
#include <libecs/FullID.hpp>
#include <libecs/Process.hpp>

LIBECS_DM_CLASS( PythonProcessBase, libecs::Process )
{
    DECLARE_ASSOCVECTOR(
        libecs::String,
        libecs::Polymorph,
        std::less< const libecs::String >,
        PropertyMap
    );

public:

    LIBECS_DM_OBJECT_ABSTRACT( PythonProcessBase )
    {
        INHERIT_PROPERTIES( libecs::Process );
    }


    PythonProcessBase()
    {
        if( ! Py_IsInitialized() )
        {
            Py_Initialize();
        }
    }

    virtual ~PythonProcessBase()
    {
        // ; do nothing
    }


    boost::python::handle<> compilePythonCode( libecs::String const& aPythonCode, 
                                               libecs::String const& aFilename,
                                               int start )
    {
        return boost::python::handle<>(
            Py_CompileString(
                const_cast< char* >( aPythonCode.c_str() ),
                const_cast< char* >( aFilename.c_str() ), start ) );
    }

    void defaultSetProperty( libecs::String const& aPropertyName,
                             libecs::Polymorph const& aValue )
    {
        PropertyMapIterator i( thePropertyMap.find( aPropertyName ) );
        if ( i == thePropertyMap.end() )
        {
            thePropertyMap.insert( std::make_pair( aPropertyName, aValue ) );
        }
        else
        {
            i->second = aValue;
        }

        theLocalNamespace[ aPropertyName ] = boost::python::object(
                boost::python::borrowed(
                    PyFloat_FromDouble( aValue.as< libecs::Real >() ) ) );
    }

    const libecs::Polymorph defaultGetProperty( libecs::String const& aPropertyName ) const
    {
        PropertyMapConstIterator aPropertyMapIterator(
                thePropertyMap.find( aPropertyName ) );

        if( aPropertyMapIterator != thePropertyMap.end() )
        {
            return aPropertyMapIterator->second;
        }
        else
        {
            THROW_EXCEPTION( libecs::NoSlot, asString() + ": property [" +
                             aPropertyName + "] is not defined" );
        }
    }

    const libecs::StringVector defaultGetPropertyList() const
    {
        libecs::StringVector aVector;

        std::transform( thePropertyMap.begin(), thePropertyMap.end(),
                std::back_inserter( aVector ),
                libecs::SelectFirst< PropertyMap::value_type >() );

        return aVector;
    }

    const libecs::PropertyAttributes
    defaultGetPropertyAttributes( libecs::String const& aPropertyName ) const
    {
        return libecs::PropertyAttributes(
                libecs::PropertySlotBase::POLYMORPH,
                true, true, true, true, true );
    }

    virtual void initialize();

protected:

    boost::python::dict     theGlobalNamespace;
    boost::python::dict     theLocalNamespace;

    PropertyMap        thePropertyMap;
};


void PythonProcessBase::initialize()
{
    Process::initialize();
    
    theGlobalNamespace.clear();

    for( libecs::VariableReferenceVectorConstIterator i(
                getVariableReferenceVector().begin() );
         i != getVariableReferenceVector().end(); ++i )
    {
        libecs::VariableReferenceCref aVariableReference( *i );

        theGlobalNamespace[ aVariableReference.getName() ] = 
                boost::python::object( boost::ref( aVariableReference ) );
    }

    // extract 'this' Process's methods and attributes
    boost::python::object aPySelfProcess(
        boost::python::ptr( static_cast< Process* >( this ) ) );

    theGlobalNamespace[ "self" ] = aPySelfProcess;

    boost::python::handle<> aMainModule(
            boost::python::borrowed( PyImport_AddModule( "__main__" ) ) );
    boost::python::handle<> aMathModule(
            boost::python::borrowed( PyImport_AddModule( "math" ) ) );

    boost::python::handle<> aMainNamespace( boost::python::borrowed(
            PyModule_GetDict( aMainModule.get() ) ) );
    boost::python::handle<> aMathNamespace( boost::python::borrowed(
            PyModule_GetDict( aMathModule.get() ) ) );

    theGlobalNamespace.update( aMainNamespace );
    theGlobalNamespace.update( aMathNamespace );
}

LIBECS_DM_INIT_STATIC( PythonProcessBase, Process );

#endif /* __PYTHONPROCESSBASE_HPP */
