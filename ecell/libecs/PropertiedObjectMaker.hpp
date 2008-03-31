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

#include "ModuleManager.hpp"

namespace libecs
{

/* *defgroup libecs_module The Libecs Module
 * This is the libecs module
 * @{
 */

class LIBECS_API PropertiedObjectMaker
{
public:
    PropertiedObjectMaker( ModuleManager* moduleManager )
        : moduleManager_( moduleManager )
    {
    }

    ~PropertiedObjectMaker()
    {
    }

    template<typename T_>
    T_* make( const String& className )
    {
        // should not use dynamic_cast<> because quering RTTI to the
        // dynamicall added class is not well defined in the C++ spec.
        return reinterpret_cast< T_* >(
                make( Type2PropertiedClassKind< T_ >::value, className ) );
    }

    PropertiedClass* make( const PropertiedClassKind& kind,
            const String& className  )
    {
        const ModuleManager::Module& mod(
                moduleManager_->getModule( className ) );
        const PropertyInterface* info(
                reinterpret_cast< const PropertyInterface *>(
                    mod.getInfo( "Interface" ) ) );
        if ( info == NULL )
        {
            THROW_EXCEPTION( UnexpectedError,
                    String( "Querying interface failed on Module " )
                    + className );
        }
        if ( info->getKind() != kind )
        {
            THROW_EXCEPTION( UnexpectedError,
                    String( "Requested module " )
                    + className + " is not a " + kind.name );
        }

        return mod.createInstance();
    }

private:
    ModuleManager* moduleManager_;
};

#define NewPropertiedObjectModule(CLASS) NewDynamicModule(PropertiedObject,CLASS)

/** @} */ //end of libecs_module

} // namespace libecs

#endif /* ___PROPERTIEDOBJECTMAKER_H___ */
