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

    DataPoint( RealCref aTime, RealCref aValue )
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
      return theValue;
    }

    RealCref getMin() const
    {
      return theValue;
    }

    RealCref getMax() const
    {
      return theValue;
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
      ;
    }

    void setMin( RealCref aReal )
    {
      ;
    }

    void setMax( RealCref aReal )
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
   
    DataPoint& operator = ( const DataPointLong& dpt5 );

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


    DataPointLong( RealCref aTime, RealCref aValue ) //constructor with 2 args
    
    {
      theTime = aTime ;
      theValue = aValue ;
      theAvg = aValue ;
      theMax = aValue ;
      theMin = aValue ;
    }

    DataPointLong( RealCref aTime, RealCref aValue, RealCref anAvg,
		RealCref aMax, RealCref aMin ) //constructor with 5 args

    {
      theTime = aTime ;
      theValue = aValue ;
      theAvg = anAvg ;
      theMax = aMax ;
      theMin = aMin ;
    }


    DataPointLong( const DataPoint& dpt) // constructor from DP2
	{
	
	
		theTime = dpt.getTime() ;
		theValue = dpt.getValue() ;
		theAvg = dpt.getAvg() ;
		theMin = dpt.getMin() ;
		theMax = dpt.getMax() ;
	

	}


    DataPointLong& operator = ( const DataPoint& dpt )
	{
		setTime( dpt.getTime() );
		setValue ( dpt.getValue() );
		setAvg ( dpt.getAvg() );
		setMin ( dpt.getMin() );
		setMax ( dpt.getMax() );
	}

    ~DataPointLong()
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

	DataPointAggregator( const DataPointLong& );


	~DataPointAggregator();

	void aggregate( const DataPointLong& );

	const DataPointLong& getData();

	void beginNextPoint();

	DataPointLong getLastPoint();

	private:
	void store( const DataPointLong& );

	bool stockpile( DataPointLong&, const DataPointLong& );
	void calculate( const DataPointLong& );
	void calculateMinMax( DataPointLong&, const DataPointLong& );

	DataPointLong Accumulator;
	DataPointLong Collector;
	DataPointLong PreviousPoint;

	public:
	void printDP( const DataPointLong& );
	void printDP(  );
	};


  //@}

} // namespace libecs


#endif /* __DATAPOINT_HPP */
