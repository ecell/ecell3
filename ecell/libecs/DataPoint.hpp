//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2001-2002 Keio University
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
// 25/03/2002


#if !defined(__DATAPOINT_HPP)
#define __DATAPOINT_HPP

#include "libecs.hpp"


#include "Polymorph.hpp"

namespace libecs
{

  /** @addtogroup logging
   *@{
   */

  /** \file */



  /**

  */

  class DataPointLong;
  class DataPoint;
// two element size datapoint

  class DataPoint 
  {


  public:

    DataPoint()
      :
      theTime ( 0.0 ),
      theValue( 0.0 )
    {
      ; // do nothing
    }

    DataPoint( RealParam aTime, RealParam aValue )
      :
      theTime ( aTime ),
      theValue( aValue )   
    {
      ; //do nothing
    }



    ~DataPoint()
    {
      ; // do nothing
    }

    const Real getTime() const
    {
      return theTime;
    }

    const Real getValue() const
    {
      return theValue;
    }

    const Real getAvg() const
    {
      return theValue;
    }

    const Real getMin() const
    {
      return theValue;
    }

    const Real getMax() const
    {
      return theValue;
    }


    void setTime( RealParam aReal )
    {
      theTime = aReal;
    }

    void setValue( RealParam aReal )
    {
      theValue = aReal;
    }

    void setAvg( RealParam aReal )
    {
      ;
    }

    void setMin( RealParam aReal )
    {
      ;
    }

    void setMax( RealParam aReal )
    {
      ;
    }

    static const size_t getElementSize()
    {
      return sizeof( Real );
    }

    static const int getElementNumber()
    {
      return 2;
    }
   
    DataPointRef operator = ( DataPointLongCref );

  protected:

    Real theTime;
    Real theValue;

  };



  class DataPointLong :
	public DataPoint
  {


  public:

    DataPointLong() //constructor with no arguments
     
    {
      theTime = 0.0 ;
      theValue = 0.0 ;
      theAvg = 0.0 ;
      theMax = 0.0 ;
      theMin = 0.0 ;

    }


    DataPointLong( RealParam aTime, RealParam aValue ) //constructor with 2 args
    
    {
      theTime = aTime ;
      theValue = aValue ;
      theAvg = aValue ;
      theMax = aValue ;
      theMin = aValue ;
    }

    DataPointLong( RealParam aTime, RealParam aValue, RealParam anAvg,
		RealParam aMax, RealParam aMin ) //constructor with 5 args

    {
      theTime = aTime ;
      theValue = aValue ;
      theAvg = anAvg ;
      theMax = aMax ;
      theMin = aMin ;
    }


    DataPointLong( DataPointCref aDataPoint) // constructor from DP2
    {
      
      
      theTime = aDataPoint.getTime() ;
      theValue = aDataPoint.getValue() ;
      theAvg = aDataPoint.getAvg() ;
      theMin = aDataPoint.getMin() ;
      theMax = aDataPoint.getMax() ;
      
      
    }
    
    
    DataPointLongRef operator = ( DataPointCref aDataPoint )
    {
      setTime( aDataPoint.getTime() );
      setValue ( aDataPoint.getValue() );
      setAvg ( aDataPoint.getAvg() );
      setMin ( aDataPoint.getMin() );
      setMax ( aDataPoint.getMax() );
    }
    
    ~DataPointLong()
    {
      ; // do nothing
    }
    
    const Real getTime() const
    {
      return theTime;
    }
    
    const Real getValue() const
    {
      return theValue;
    }
    
    const Real getAvg() const
    {
      return theAvg;
    }
    
    const Real getMin() const
    {
      return theMin;
    }

    const Real getMax() const
    {
      return theMax;
    }


    void setTime( RealParam aReal )
    {
      theTime = aReal;
    }

    void setValue( RealParam aReal )
    {
      theValue = aReal;
    }

    void setAvg( RealParam aReal )
    {
      theAvg = aReal;
    }

    void setMin( RealParam aReal )
    {
      theMin = aReal;
    }

    void setMax( RealParam aReal )

    {
      theMax = aReal;
    }

    static const size_t getElementSize()
    {
      return sizeof( Real);
    }

    static const int getElementNumber()
    {
      return 5;
    }
   
  protected:

    Real theAvg;
    Real theMax;
    Real theMin;

  };



  class DataPointAggregator
  {
    
  public:
    
    DataPointAggregator();
    
    DataPointAggregator( DataPointLongCref );
    
    
    ~DataPointAggregator();
    
    void aggregate( DataPointLongCref );
    
    DataPointLongCref getData();
    
    void beginNextPoint();
    
    DataPointLong getLastPoint();
    
  private:
    void store( DataPointLongCref );
    
    bool stockpile( DataPointLongRef, DataPointLongCref );
    void calculate( DataPointLongCref );
    void calculateMinMax( DataPointLongRef, DataPointLongCref );
    
    DataPointLong theAccumulator;
    DataPointLong theCollector;
    DataPointLong thePreviousPoint;
    
  };
  
  
  //@}
  
} // namespace libecs


#endif /* __DATAPOINT_HPP */
