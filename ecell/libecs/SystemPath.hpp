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

#ifndef SYSTEMPATH_HPP
#define SYSTEMPATH_HPP

#include <vector>
#include "libecs.hpp"

/** @addtogroup identifier The FullID, FullPN and SystemPath.
 The FullID, FullPN and SystemPath.
 

 @ingroup libecs
 @{
 */

/** @file */


namespace libecs {

/**
 SystemPath
 */
class LIBECS_API SystemPath : protected std::vector<String>
{
public:
    typedef std::vector<String> Base;
public:
    SystemPath( const Base& systempath )
            : Base( systempath )
    {
        ; // do nothing
    }

    SystemPath(): Base()
    {
    }

    ~SystemPath() {}

    const String asString() const;

    bool isAbsolute() const
    {
        return ( ( ( ! empty() ) && ( front()[0] == DELIMITER ) ) || empty() );
    }

    /**
       Normalize a SystemPath.
       Reduce '..'s and remove extra white spaces.

       @return reference to the systempath
    */
    SystemPath normalize();

    LIBECS_API static SystemPath parse( const String& systempathstring );

    bool operator==(const SystemPath& rhs) const
    {
        return static_cast<const Base&>(*this) == static_cast<const Base&>(rhs);
    }

    bool operator!=(const SystemPath& rhs) const
    {
        return static_cast<const Base&>(*this) != static_cast<const Base&>(rhs);
    }

    bool operator<(const SystemPath& rhs) const
    {
        return static_cast<const Base&>(*this) < static_cast<const Base&>(rhs);
    }

public:
    static const char DELIMITER = '/';
};

} // namespace libecs

/** @} */ // identifier module

#endif // SYSTEMPATH_HPP

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
