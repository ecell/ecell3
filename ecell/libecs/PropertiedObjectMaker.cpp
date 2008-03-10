//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell PropertiedObjectMaker
//
//       Copyright (C) 1996-2008 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell PropertiedObjectMaker is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
//
// E-Cell PropertiedObjectMaker is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public
// License along with E-Cell PropertiedObjectMaker -- see the file COPYING.
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

#include "PropertiedObjectMaker.hpp"

namespace libecs
{
PropertiedObjectMaker::PropertiedObjectMaker()
        : theModuleMakerList()
{
}

PropertiedObjectMaker::~PropertiedObjectMaker()
{
    ; // do nothing
}

void
PropertiedObjectMaker::addModuleMaker( ModuleMaker* maker )
{
    theModuleMakerList.add( maker );
}

void
PropertiedObjectMaker::removeModuleMaker( ModuleMaker* maker )
{
    theModuleMakerList.remove( maker );
}

const Module&
PropertiedObjectMaker::getModule( const String& name ) const
{
    for ( ModuleMakerList::const_iterator i( theModuleMakerList.begin() );
            i != theModuleMakerList.end(); ++i )
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

    return Module( name ); // never get here
}

} // namespace libecs
