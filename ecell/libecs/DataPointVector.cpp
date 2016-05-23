//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2016 Keio University
//       Copyright (C) 2008-2016 RIKEN
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
// 14/04/2002

#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include "DataPointVector.hpp"
#include <cassert>

namespace libecs
{

DataPointVector::~DataPointVector()
{
    if( thePointSize==2 )
    {
        delete[] theRawArray;
    }
    else
    {
        delete[] theRawArrayLong;
    }
}

DataPointVector::DataPointVector( std::size_t aLength, int aPointSize ) 
    : theSize( aLength ),
      thePointSize( aPointSize )
{
    // init the appropriate array

    if( thePointSize == 2 )
    {
        theRawArray = new DataPoint[ aLength ];
    }
    else
    {
        theRawArrayLong = new LongDataPoint[ aLength ];
    }
}


size_t DataPointVector::getElementSize() const
{
    if( thePointSize == 2 )
    {
        return sizeof(DataPoint);
    }
    return sizeof(LongDataPoint);
}


DataPoint& DataPointVector::asShort( std::size_t aPosition )
{
    assert (thePointSize == 2);
    return theRawArray[ aPosition ];
}

DataPoint const& 
DataPointVector::asShort( std::size_t aPosition ) const
{
    assert (thePointSize == 2);
    return theRawArray[ aPosition ];
}

LongDataPoint& DataPointVector::asLong( std::size_t aPosition )
{
    assert (thePointSize == 5);
    return theRawArrayLong[ aPosition ];
}


LongDataPoint const& 
DataPointVector::asLong( std::size_t aPosition ) const
{
    assert (thePointSize == 5);
    return theRawArrayLong[ aPosition ];
}


const void* DataPointVector::getRawArray() const
{
    if (thePointSize == 2)
    {
        return (void*) theRawArray;
    }
    return (void*) theRawArrayLong;

}

Integer DataPointVector::getPointSize()
{
    return DataPointVector::thePointSize;
}

} // namespace libecs
