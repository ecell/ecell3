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
// written by Masayuki Okayama <smash@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//
// modified by Gabor Bereczki <gabor.bereczki@talk21.com>
// 14/04/2002


#include "Logger.hpp"

 

namespace libecs
{

  // Constructor
  Logger::Logger( ModelCref aModel, PropertySlotRef aPropertySlot )
    :
    theModel( aModel ),
    thePropertySlot( aPropertySlot ),
    theMinimumInterval( 0.0 ),
    theCurrentInterval( 0.0 )
  {
    ; // do nothing
  }


  DataPointVectorRCPtr Logger::getData( void ) 
  {
    theDataPointVector 
      = thePhysicalLogger.getVector( thePhysicalLogger.begin(),
				     thePhysicalLogger.end() );
    
    return theDataPointVector;
  }

  //

  DataPointVectorRCPtr Logger::getData( RealCref aStartTime,
					RealCref anEndTime ) 
  {
    PhysicalLoggerIterator 
      top( thePhysicalLogger.upper_bound( thePhysicalLogger.begin(),
					  thePhysicalLogger.end(), 
					  anEndTime ) );

    PhysicalLoggerIterator 
      bottom( thePhysicalLogger.lower_bound( thePhysicalLogger.begin(),
					     top,
					     aStartTime ) );

    theDataPointVector = thePhysicalLogger.getVector( bottom, top );

    return theDataPointVector;
  }


  //


  DataPointVectorRCPtr Logger::getData( RealCref aStartTime,
					RealCref anEndTime,
					RealCref anInterval ) 
  {
    DataPointVectorIterator 
      range( static_cast<DataPointVectorIterator>
	     ( ( anEndTime - aStartTime ) / anInterval ) + 1 );
    DataPointVectorIterator counter( 0 );

    PhysicalLoggerIterator 
      top( thePhysicalLogger.upper_bound( thePhysicalLogger.begin(),
					  thePhysicalLogger.end(),
					  anEndTime ) );

    PhysicalLoggerIterator 
      bottom( thePhysicalLogger.lower_bound( thePhysicalLogger.begin(),
					     top,
					     aStartTime ) );

    Real rcounter( aStartTime );
    DataPointVectorPtr aDataPointVector( new DataPointVector( range ) );
    DataPoint aDataPoint;
    PhysicalLoggerIterator it;
    while( counter < range )
      {
	it = thePhysicalLogger.lower_bound( bottom, top , rcounter );
	thePhysicalLogger.getItem( it, &aDataPoint );
	( *aDataPointVector )[ counter ] = aDataPoint;
	++counter;
	rcounter += anInterval;
      }

    theDataPointVector = aDataPointVector;

    return theDataPointVector;
  }


  void Logger::appendData( RealCref aValue )
  {
    const Real aTime( theModel.getCurrentTime() );

    if( !thePhysicalLogger.empty() )
      {
    	theCurrentInterval = aTime - thePhysicalLogger.back().getTime();
      }

    thePhysicalLogger.push( aTime, aValue );
    if( theMinimumInterval < theCurrentInterval )
      {
	theMinimumInterval = theCurrentInterval;
      }
  }

  //

  StringCref Logger::getName() const
  {
    return thePropertySlot.getName();
  }


  //

  Real Logger::getStartTime( void ) 
  {
    if( !thePhysicalLogger.empty() )
      {
	return thePhysicalLogger.front().getTime();
      }
    else
      {
	static const Real aZero( 0.0 );
	return aZero;
      }
  }


  //

  Real Logger::getEndTime( void ) 
  {
    if( !thePhysicalLogger.empty() )
      {
	return thePhysicalLogger.back().getTime();
      }
    else
      {
	static const Real aZero( 0.0 );
	return aZero;
      }
  }



} // namespace libecs


#ifdef LOGGER_TEST


#include <stdio.h>
#include <iostream>
#include "DataPoint.cpp"
#include "StlDataPointVector.cpp"

using namespace libecs;

const Real& func(void)
{
  const Real* fp = new Real(3.14);
  return *fp;
}

main()
{


  Logger<Real,Real> lg = Logger<Real,Real>(op);
  /*
    lg.update(d1);
    lg.update(d2);
    lg.update(d3);
    lg.update(d4);
    lg.update(d5);
  */


  Logger<Real,Real> lg_clone = Logger<Real,Real>(lg);

  //  printf("%p %p\n",&(lg.getDataPointVector()),&(lg_clone.getDataPointVector()));

  lg.getData(0.0,5.0,0.5);
  printf("%f\n",lg.getStartTime());
  printf("%f\n",lg.getEndTime());
}



#endif /* LOGGER_TEST */
