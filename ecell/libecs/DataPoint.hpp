//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 2001-2002 Keio University
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
// 25/03/2002


#if !defined(__DATAPOINT_HPP)
#define __DATAPOINT_HPP

#include "libecs.hpp"


#include "UVariable.hpp"

namespace libecs
{

  /** @defgroup libecs_module The Libecs Module 
   * This is the libecs module 
   * @{ 
   */ 
  

  class DataPoint
  {

  public:

    DataPoint( RealCref aTime = 0.0, RealCref aValue = 0.0 )
      :
      theTime( aTime ),
      theValue( aValue )
    {
      ; // do nothing
    }

    DataPoint( DataPointCref aDataPoint )
    {
      theTime  = aDataPoint.getTime();
      theValue = aDataPoint.getValue();
    }

    DataPoint( DataPointRef aDataPoint )
    {
      theTime  = aDataPoint.getTime();
      theValue = aDataPoint.getValue();
    }


    ~DataPoint()
    {
      ; // do nothing
    }

    RealCref getTime() const
    {
      return theTime;
    }

    RealCref getValue() const
    {
      return theValue;
    }

    void SetValue( RealCref aValue )
    {
      theValue = aValue;
    }

    void SetTime( RealCref aTime )
    {
      theTime = aTime;
    }

    DataPointRef operator=( DataPointCref aDataPoint)
    {
      SetTime(  aDataPoint.getTime() );
      SetValue( aDataPoint.getValue() );

      return *this;
    }

    DataPointRef operator=( DataPointRef aDataPoint )
    {
      SetTime(  aDataPoint.getTime() );
      SetValue( aDataPoint.getValue() );

      return *this;
    }


  private:

    Real theTime;
    Real theValue;

  };


  /** @} */ //end of libecs_module 

} // namespace libecs


#endif /* __DATAPOINT_HPP */
