//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2010 Keio University
//       Copyright (C) 2005-2009 The Molecular Sciences Institute
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
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#ifndef ___ECSOBJECTMAKER_H
#define ___ECSOBJECTMAKER_H

#include "dmtool/ModuleMaker.hpp"
#include "dmtool/DynamicModuleInfo.hpp"

#include "libecs/EcsObject.hpp"
#include "libecs/PropertyInterface.hpp"

namespace libecs
{

class Stepper;
class Variable;
class Process;
class System;

template< typename T_ >  
class EcsObjectMaker 
{
public:
    typedef ModuleMaker< EcsObject > Backend;
    typedef T_ DMType;

public:
    EcsObjectMaker( Backend& aBackend )
      : theBackend( aBackend )
    {
        // do nothing;
    }

    static const char* getTypeName();

    /**
       Instantiates given class of an object.

       @param classname name of class to be instantiated.
       @return pointer to a new instance.
    */
    DMType* make( const std::string& aClassname )
    {
        DMType* anInstance( reinterpret_cast< DMType *>( getModule( aClassname ).createInstance() ) );
        if ( !anInstance )
        {
            THROW_EXCEPTION( Instantiation,
                             "cannot instantiate [" + aClassname + "]" );
        }

        return anInstance;
    }

    const Backend::Module& getModule(String const& aClassName )
    {
        const Backend::Module& mod( theBackend.getModule( aClassName ) );
        const PropertyInterface< DMType >* info(
                reinterpret_cast< const PropertyInterface< DMType >* >( mod.getInfo() ) );
        if ( !info || info->getTypeName() != getTypeName() )
        {
            THROW_EXCEPTION( TypeError,
                    String( "Specified class [" + aClassName + "] is not a " )
                    + getTypeName() );
        }
        return mod;
    }

protected:
    Backend& theBackend;
};

template<>  
inline const char* EcsObjectMaker<Stepper>::getTypeName()
{
    return "Stepper";
}

template<>  
inline const char* EcsObjectMaker<Process>::getTypeName()
{
    return "Process";
}

template<>  
inline const char* EcsObjectMaker<Variable>::getTypeName()
{
    return "Variable";
}

template<>  
inline const char* EcsObjectMaker<System>::getTypeName()
{
    return "System";
}

} // namespace libecs

#endif /* ___ECSOBJECTMAKER_H */
