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
#include "FullPN.hpp"

namespace libecs
{

FullPN FullPN::parse( const String& fullpropertynamestring )
{
    String::size_type aPosition( 0 );

    for ( int i( 0 ) ; i < 3 ; ++i )
    {
        aPosition = fullpropertynamestring.
                    find_first_of( FullID::DELIMITER, aPosition );
        if ( aPosition == String::npos )
        {
            THROW_EXCEPTION( BadFormat,
                             "not enough fields in FullPN string \"" +
                             fullpropertynamestring + "\"" );
        }
        ++aPosition;
    }

    return FullPN(
           FullID::parse(
               fullpropertynamestring.substr( 0, aPosition - 1 ) ),
           fullpropertynamestring.substr( aPosition ) );
}

const String FullPN::asString() const
{
    return fullID_.asString() + FullID::DELIMITER + propertyName_;
}

} // namespace libecs

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
