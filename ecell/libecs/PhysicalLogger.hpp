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
//


#if !defined(__PHYSICALLOGGER_HPP)
#define __PHYSICALLOGGER_HPP

#include "Exceptions.hpp"
#include "VVector.h"
#include "DataPoint.hpp"

#include "libecs.hpp"
#include "DataPointVector.hpp"

namespace libecs
{


  /** @addtogroup logging
   *@{
  */

  /** @file */



  class PhysicalLogger
  {

    typedef vvector<DataPoint> Vector;
    typedef vvector<DataPointLong> VectorLong;
    
  public:

    DECLARE_TYPE( Vector::size_type, iterator );
    DECLARE_TYPE( Vector::size_type, size_type );

    PhysicalLogger(int );

    virtual ~PhysicalLogger()
    {
      ; // do nothing
    }
	
    void push( DataPointCref  );

    void push( DataPointLongCref  );

    void aggregate( DataPointCref );

    void aggregate( DataPointLongCref );

    void flushAggregate();

    DataPointLongCref getAggregate()
    {
      return theAggregator.getData();
    }

    int getElementCount()
    {
      return theElementCount;
    }

    void resetElementCount()
    {
      theElementCount = 0;
    }

    void setEndPolicy( Integer );

    void setMaxSize( iterator );


    iterator lower_bound( const iterator& start,
			  const iterator& end,
			  const Real time ) const;

    iterator upper_bound( const iterator& start,
			  const iterator& end,
			  const Real time ) const;

    iterator lower_bound_linear( const iterator& start,
				 const iterator& end,
				 const Real time ) const;

    iterator upper_bound_linear( const iterator& start,
				 const iterator& end,
				 const Real time ) const;

    iterator lower_bound_linear_backwards( const iterator& start,
					   const iterator& end,
					   const Real time ) const;

    iterator lower_bound_linear_estimate( const iterator& start,
					  const iterator& end,
					  const Real time,
					  const Real time_per_step ) const;

    iterator upper_bound_linear_estimate( const iterator& start,
					  const iterator& end,
					  const Real time,
					  const Real time_per_step ) const;
    
    iterator next_index( const iterator& start) const;

    void getItem( const iterator&, DataPointPtr ) const;
    
    DataPointVectorSharedPtr getVector( const iterator& start,
					const iterator& end ) const;

    size_type size() const;

    bool empty() const;

    DataPointLong front() const;

    DataPointLong back() const;

    iterator begin() const;

    iterator end() const;

    Real getAverageInterval() const;

    DataPointLong at( const iterator& ) const;

  private:
    iterator            theCurrentPosition;
    DataPoint           anEmptyDataPoint;

    // this mutable can be removed if vvector supports const operations
    mutable Vector      theVector;
    mutable VectorLong  theVectorLong;
    Integer             PointSize;
    DataPointAggregator theAggregator;
    int                 theElementCount;
    
  };


  DECLARE_TYPE( PhysicalLogger::iterator, PhysicalLoggerIterator );


  //@}

} // namespace libecs


#endif /* __PHYSICALLOGGER_HPP */
