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

#include "VVector.h"
#include "DataPoint.hpp"

#include "libecs.hpp"
#include "DataPointVector.hpp"

namespace libecs
{

  /** @defgroup libecs_module The Libecs Module 
   * This is the libecs module 
   * @{ 
   */ 
  

  class PhysicalLogger
  {

    typedef vvector<DataPoint> Vectortype;
    
  public:

    DECLARE_TYPE( Vectortype::size_type, iterator );
    DECLARE_TYPE( Vectortype::size_type, size_type );

    PhysicalLogger();

    virtual ~PhysicalLogger()
    {
      ; // do nothing
    }
	

    void push( RealCref aTime, RealCref aValue )
    {
      DataPoint aDataPoint( aTime, aValue );
      theVector.push_back( aDataPoint );
    }

    iterator lower_bound( const iterator& start,
			  const iterator& end,
			  RealCref time );

    iterator upper_bound( const iterator& start,
			  const iterator& end,
			  RealCref time );

    void getItem( const iterator&, DataPointPtr );

    DataPointVectorRCPtr getVector( const iterator& start,
				    const iterator& end );

    size_type size() const
    {
      return theVector.size();
    }


    bool empty() const
    {
      return ( size() == 0 );
    }

    DataPoint front()
    {
      return theVector[ 0 ];
    }

    DataPoint back()
    {
      // danger!!  undefined behavior with vvector if size() == 0 - sha
      return theVector[ size() - 1 ];
    }


    iterator begin() const
    {
      return 0;
    }

    iterator end() const
    {
      // is this ok? - sha
      return size();
    }



  private:

    iterator theCurrentPosition;

    DataPointVectorRCPtr theDPVector;

    Vectortype theVector;

  };


  DECLARE_TYPE( PhysicalLogger::iterator, PhysicalLoggerIterator );


  /** @} */ //end of libecs_module 

} // namespace libecs


#endif /* __PHYSICALLOGGER_HPP */
