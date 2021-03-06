//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2014 Keio University
//       Copyright (C) 2008-2014 RIKEN
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
// written by Gabor Bereczki <gabor.bereczki@talk21.com>
//


#if !defined( __DATAPOINTVECTOR_HPP )
#define __DATAPOINTVECTOR_HPP

#include "libecs/Defs.hpp"
#include "libecs/DataPoint.hpp"
#include <sys/types.h>

namespace libecs
{

class LIBECS_API DataPointVector
{
public:
    DataPointVector( std::size_t, int aPointSize );

    ~DataPointVector();

    DataPoint& asShort( std::size_t aPosition );

    DataPoint const& asShort( std::size_t aPosition ) const;

    LongDataPoint& asLong( std::size_t aPosition );

    LongDataPoint const& asLong( std::size_t aPosition ) const;

    std::size_t getSize() const
    {
        return theSize;
    }

    std::size_t getElementSize() const;
            
    std::size_t begin() const
    {
        return 0;
    }
            
    std::size_t end() const
    {
        return getSize();
    }
            
    const void* getRawArray() const;

    Integer getPointSize();

private:
    std::size_t theSize;

    Integer thePointSize;
    
    DataPoint* theRawArray;

    LongDataPoint* theRawArrayLong;
};

} // namespace libecs

#endif /* __DATAPOINTVECTOR_HPP */
