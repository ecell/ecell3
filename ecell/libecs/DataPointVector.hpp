//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 2002-2002 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-CELL is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
//
// E-CELL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public
// License along with E-CELL -- see the file COPYING.
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


  class DataPointVector
  {

  public:

    DataPointVector( DataPointVectorIterator, Int );

    ~DataPointVector();

    DataPointRef asShort( DataPointVectorIterator aPosition );

    DataPointCref asShort( DataPointVectorIterator aPosition ) const;

    DataPointLongRef asLong( DataPointVectorIterator aPosition );

    DataPointLongCref asLong( DataPointVectorIterator aPosition ) const;

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

  private:

    DataPointVectorIterator theSize;

	Int PointSize;
    
    DataPoint* theRawArray;

	DataPointLong* theRawArrayLong;

  };

  //@}

} // namespace libecs


#endif /* __DATAPOINTVECTOR_HPP */
