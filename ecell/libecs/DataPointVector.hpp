//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2002-2002 Keio University
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
//


#if !defined( __DATAPOINTVECTOR_HPP )
#define __DATAPOINTVECTOR_HPP

#include "DataPoint.hpp"
#include "Defs.hpp"
#include <sys/types.h>


namespace libecs
{

  /** @addtogroup logging
   *@{
   */

  /** @file */


  DECLARE_TYPE( size_t, DataPointVectorIterator );

  DECLARE_SHAREDPTR( DataPointVector );

  class DataPointVector
  {

  public:

    DataPointVector( DataPointVectorIterator, int aPointSize );

    ~DataPointVector();

    ECELL_API DataPointRef asShort( DataPointVectorIterator aPosition );

    DataPointCref asShort( DataPointVectorIterator aPosition ) const;

    ECELL_API LongDataPointRef asLong( DataPointVectorIterator aPosition );

    LongDataPointCref asLong( DataPointVectorIterator aPosition ) const;

    DataPointVectorIterator getSize() const
    {
      return theSize;
    }

    size_t getElementSize() const;
	
    DataPointVectorIterator begin() const
    {
      return 0;
    }
	
    DataPointVectorIterator end() const
    {
      return getSize();
    }
	
    const void* getRawArray() const;

    ECELL_API Integer getPointSize();

  private:

    DataPointVectorIterator theSize;

    Integer thePointSize;
    
    DataPoint* theRawArray;

    LongDataPoint* theRawArrayLong;

  };

  //@}

} // namespace libecs


#endif /* __DATAPOINTVECTOR_HPP */
