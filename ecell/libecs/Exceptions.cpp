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
#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include "Exceptions.hpp"

namespace libecs
{

Exception::~Exception() throw()
{
    ; // do nothing
}

// not really a const function as it does the lazy initialization
const String& Exception::asString() const
{
    if ( !theStringRepr.empty() )
        return theStringRepr;

    String tmp;
    tmp += getClassName();
    if ( !theMethod.empty() )
    {
        tmp += " (";
        tmp += theMethod;
        tmp += ")";
    }
    tmp += ": ";
    tmp += theMessage;
    const_cast<Exception*>(this)->theStringRepr.swap( tmp );
    return theStringRepr;
}

} // namespace libecs
