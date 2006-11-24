//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2000-2001 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
//
// E-Cell is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public
// License along with E-Cell -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
//
//END_HEADER
//
// written by Gabor Bereczki <gabor.bereczki@talk21.com>
// 14/04/2002


#include "DataPointVector.hpp"
#include <assert.h>


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



  DataPointVector::DataPointVector( DataPointVectorIterator aLength, 
				    int aPointSize ) 
    :
    theSize( aLength ),
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





  DataPointRef DataPointVector::asShort( DataPointVectorIterator aPosition )
  {
    assert (thePointSize == 2);
    return theRawArray[ aPosition ];
  }

  DataPointCref 
  DataPointVector::asShort( DataPointVectorIterator aPosition ) const
  {
    assert (thePointSize == 2);
    return theRawArray[ aPosition ];
  }

  LongDataPointRef DataPointVector::asLong( DataPointVectorIterator aPosition )
  {
    assert (thePointSize == 5);
    return theRawArrayLong[ aPosition ];
  }


  LongDataPointCref 
  DataPointVector::asLong( DataPointVectorIterator aPosition ) const
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


