//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell ModuleManager
//
//       Copyright (C) 1996-2008 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell ModuleManager is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
//
// E-Cell ModuleManager is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public
// License along with E-Cell ModuleManager -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
//
//END_HEADER
//
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//
#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include "ModuleManager.hpp"

namespace libecs
{
ModuleManager::ModuleManager()
        : moduleMakers_()
{
}

ModuleManager::~ModuleManager()
{
    ; // do nothing
}

void
ModuleManager::addModuleMaker( ModuleMaker* maker )
{
    moduleMakers_.insert( maker );
}

void
ModuleManager::removeModuleMaker( ModuleMaker* maker )
{
    moduleMakers_.erase( maker );
}

const ModuleManager::Module&
ModuleManager::getModule( const String& name ) const
{
    for ( ModuleMakerSet::const_iterator i( moduleMakers_.begin() );
            i != moduleMakers_.end(); ++i )
    {
        try
        {
            const Module& retval( ( *i )->getModule( name, false ) );
            return retval;
        }
        catch ( const DMException& e )
        {
            ; // do nothing
        }
    }
    THROW_EXCEPTION( NotFound,
                     "Class [ " + name + "] not found"
                   );
}

} // namespace libecs
