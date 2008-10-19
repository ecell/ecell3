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
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#ifndef ___PROPERTIEDOBJECTMAKER_H___
#define ___PROPERTIEDOBJECTMAKER_H___

#include "dmtool/ModuleMaker.hpp"
#include "dmtool/DynamicModuleInfo.hpp"

#include "libecs/PropertiedClass.hpp"
#include "libecs/PropertyInterface.hpp"

namespace libecs
{
  
/* *defgroup libecs_module The Libecs Module 
* This is the libecs module 
* @{ 
*/ 

template< typename T_ >  
class PropertiedObjectMaker 
{
public:
    typedef StaticModuleMaker< PropertiedClass > Backend;
    typedef T_ DMType;

public:
    PropertiedObjectMaker( Backend& aBackend )
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
        Backend::Module::DMAllocator anAllocator( getModule( aClassname ).getAllocator() );
        if ( !anAllocator )
        {
            THROW_EXCEPTION( Instantiation, "Unexpected error" );
        }

        DMType* anInstance( reinterpret_cast< DMType *>( anAllocator() ) );
        if ( !anInstance )
        {
            THROW_EXCEPTION( Instantiation, "Can't instantiate [" + aClassname + "]." );
        }

        return anInstance;
    }

    const Backend::Module& getModule(StringCref aClassName )
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
inline const char* PropertiedObjectMaker<Stepper>::getTypeName()
{
    return "Stepper";
}

template<>  
inline const char* PropertiedObjectMaker<Process>::getTypeName()
{
    return "Process";
}

template<>  
inline const char* PropertiedObjectMaker<Variable>::getTypeName()
{
    return "Variable";
}

template<>  
inline const char* PropertiedObjectMaker<System>::getTypeName()
{
    return "System";
}

#define NewPropertiedObjectModule(CLASS) NewDynamicModule(PropertiedObject,CLASS)

/** @} */ //end of libecs_module 

} // namespace libecs

#endif /* ___PROPERTIEDOBJECTMAKER_H___ */
