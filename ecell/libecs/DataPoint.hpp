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

    DataPoint()
      :
      theTime ( 0.0 ), 
      theValue( 0.0 ), 
      theAvg  ( 0.0 ), 
      theMin  ( 0.0 ),
      theMax  ( 0.0 )      
    {
      ; // do nothing
    }

    DataPoint( RealCref aTime, RealCref aValue, 
	       RealCref anAvg, RealCref aMin, RealCref aMax )
      : 
      theTime ( aTime ), 
      theValue( aValue ), 
      theAvg  ( anAvg ), 
      theMin  ( aMin ),
      theMax  ( aMax )      
    {
      ; //do nothing
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

    RealCref getAvg() const
    {
      return theAvg;
    }

    RealCref getMin() const
    {
      return theMin;
    }

    RealCref getMax() const
    {
      return theMax;
    }


    void setTime( RealCref aReal )
    {
      theTime = aReal;
    }

    void setValue( RealCref aReal )
    {
      theValue = aReal;
    }

    void setAvg( RealCref aReal )
    {
      theAvg = aReal;
    }

    void setMin( RealCref aReal )
    {
      theMin = aReal;
    }

    void setMax( RealCref aReal )
    {
      theMax = aReal;
    }


    /*

    no need to define these; default methods should be most efficient

    DataPoint( DataPointCref aDataPoint )
    {
      copy( aDataPoint );
    }

    DataPoint( DataPointRef aDataPoint )
    {
      copy( aDataPoint );
    }

    DataPointRef operator=( DataPointCref aDataPoint )
    {
      copy( aDataPoint );

      return *this;
    }

    DataPointRef operator=( DataPointRef aDataPoint )
    {
      copy( aDataPoint );

      return *this;
    } 
    */ 
	
    static const size_t getElementSize()
    {
      return sizeof( Real );
    }
    
  protected:

    Real theTime;
    Real theValue;
    Real theAvg;
    Real theMin;
    Real theMax;

  public:

    enum 
      {  
	TIME_OFFSET = 0,
	VALUE_OFFSET,
	AVG_OFFSET,
	MIN_OFFSET,
	MAX_OFFSET,
	DATAPOINT_LENGTH 
      };


  };


  class DataInterval 
  {
    
  public:

    DataInterval() 
      : 
      theDataPoint(),
      theInterval( -1 )
    {
      ; //do nothing
    }

    DataPointCref getDataPoint() const
    {
      return theDataPoint;
    }

    DataPointRef getDataPoint()
    {
      return theDataPoint;
    }

    Real getInterval() const
    {
      return theInterval;
    }

    RealCref getTime() const
    {
      return getDataPoint().getTime();
    }

    RealCref getValue() const
    {
      return getDataPoint().getValue();
    }

    RealCref getAvg() const
    {
      return getDataPoint().getAvg();
    }

    RealCref getMin() const
    {
      return getDataPoint().getMin();
    }

    RealCref getMax() const
    {
      return getDataPoint().getMax();
    }

    void beginNewInterval()
    {
      theInterval = -1.0;
    }
    
    void addPoint( RealCref aTime, RealCref aValue ) 
    {
      if ( theInterval < 0 ) //the very first time data is added
	{
	  theDataPoint.setTime ( aTime );
	  theDataPoint.setValue( aValue );
	  theDataPoint.setAvg  ( aValue );
	  theDataPoint.setMin  ( aValue );
	  theDataPoint.setMax  ( aValue );
	  theInterval = 0.0;    
	}
      else
	{
	  const Real aNewInterval( aTime - getTime() );

	  theDataPoint.setAvg( getAvg() * theInterval + 
			       aValue * aNewInterval );
	  theInterval += aNewInterval;
	  theDataPoint.setAvg( getAvg() / getInterval() );

	  if ( aValue < getMin() ) 
	    { 
	      theDataPoint.setMin( aValue ); 
	    }
	  else if ( aValue > getMax() )  // assume theMax > theMin
	    { 
	      theDataPoint.setMax( aValue );
	    }

	  theDataPoint.setValue( aValue );
	  theDataPoint.setTime( aTime );
	}
    }

    void aggregatePoint( DataPointCref aDataPoint, RealCref anInterval )
    {
      if ( theInterval < 0 ) //the very first time data is added
	{
	  theDataPoint = aDataPoint;
	  theInterval = anInterval;
	}
      else
	{
	  theDataPoint.setAvg( getAvg() * getInterval() + 
			       aDataPoint.getAvg() * anInterval );
	  theInterval += anInterval;
	  theDataPoint.setAvg( getAvg() / getInterval() );
	  
	  if( aDataPoint.getMin() < getMin() ) 
	    { 
	      theDataPoint.setMin( aDataPoint.getMin() );
	    }
	  if ( aDataPoint.getMax() > getMax() ) 
	    { 
	      theDataPoint.setMax( aDataPoint.getMax() );
	    }
	  
	  theDataPoint.setValue( aDataPoint.getValue() );
	  theDataPoint.setTime( getTime() + anInterval );
	}
    }
  
    private:

      DataPoint theDataPoint;
      Real      theInterval;

    };


    /** @} */ //end of libecs_module 

  } // namespace libecs


#endif /* __DATAPOINT_HPP */
