//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 2000-2001 Keio University
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
    
  public:

    DECLARE_TYPE( Vector::size_type, iterator );
    DECLARE_TYPE( Vector::size_type, size_type );

    PhysicalLogger();

    virtual ~PhysicalLogger()
    {
      ; // do nothing
    }
	
    void push( DataPointCref aDataPoint )
    {
      theVector.push_back( aDataPoint );
    }

    iterator lower_bound( const iterator& start,
			  const iterator& end,
			  RealCref time ) const;

    iterator upper_bound( const iterator& start,
			  const iterator& end,
			  RealCref time ) const;

    void getItem( const iterator&, DataPointPtr ) const;

    DataPointVectorRCPtr getVector( const iterator& start,
				    const iterator& end ) const;

    size_type size() const
    {
      return theVector.size();
    }


    bool empty() const
    {
      return ( size() == 0 );
    }

    DataPoint front() const
    {
      if ( empty ( ) )
        {
    	    return anEmptyDataPoint;
	}
	return theVector[ 0 ];
    }

    DataPoint back() const
    {
      // danger!!  undefined behavior with vvector if size() == 0 - sha
//      DEBUG_EXCEPTION( size() > 0, AssertionFailed, "" );
//	not anymore - gabor
      if ( empty ( ) )
        {
    	    return anEmptyDataPoint;
	}

      return theVector[ end() ];
    }

    iterator begin() const
    {
      return 0;
    }

    iterator end() const
    {
      // is this ok? - sha
      //no, but I fixed it - gabor
       if ( size() > 0 )
        {
    	    return size() - 1;
	}
       else
        {
	    return 0;
	}
    }



  private:

    iterator theCurrentPosition;
    DataPoint anEmptyDataPoint;

    // this mutable can be removed if vvector supports const operations
    mutable Vector theVector;
    
  };


  DECLARE_TYPE( PhysicalLogger::iterator, PhysicalLoggerIterator );


  //@}

} // namespace libecs


#endif /* __PHYSICALLOGGER_HPP */
