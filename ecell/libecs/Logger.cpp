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

#include <iostream>

#include "PropertyInterface.hpp"

/*

 */

#if !defined(__LOGGER_HPP)
#include "Logger.hpp"
#endif



namespace libecs
{

  // Constructor

  Logger::Logger( const GetCurrentTimeMethodType& aGetCurrentTime,
		  PropertySlotPtr aPropertySlot )
    :
    thePropertySlot( aPropertySlot ),
    theGetCurrentTimeMethod( aGetCurrentTime ),
    theMinimumInterval( 0.0 ),
    theCurrentInterval( 0.0 )
  {
    ; // do nothing
  } 
  
  
  Logger::DataPointVectorCref Logger::getData( void ) const
  {
    return theDataPointVector;
  }
  
  //
  
  const Logger::DataPointVector Logger::getData( RealCref aStartTime,
						 RealCref anEndTime ) const
  {
    const const_iterator aHeadItr( theDataPointVector.begin() );
    const const_iterator aTailItr( theDataPointVector.end() );
    
    const const_iterator 
      aStartIterator( theDataPointVector.lower_bound( aHeadItr,
						      aTailItr,
						      aStartTime ) );
    const const_iterator 
      anEndIterator( theDataPointVector.upper_bound( aStartIterator,
						     aTailItr,
						     anEndTime ) );

    DataPointVector aNewDataPointVector( aStartIterator, anEndIterator );

    return aNewDataPointVector;
  }
  
  
  //
  
  
  const Logger::DataPointVector Logger::getData( RealCref aStartTime,
						 RealCref anEndTime,
						 RealCref anInterval ) const
  {
    const const_iterator aHeadIterator( theDataPointVector.begin() );
    const const_iterator aTailIterator( theDataPointVector.end() - 1 );
    
    const_iterator 
      aStartIterator( theDataPointVector.lower_bound( aHeadIterator,
						      aTailIterator,
						      aStartTime ) );
    const_iterator 
      anEndIterator( theDataPointVector.upper_bound( aStartIterator,
						     aTailIterator,
						     anEndTime ) );    

    Real aTime( aStartIterator->getTime() );
    const Real aLastTime( anEndIterator->getTime() );

    DataPointVector aDataPointVector;
    aDataPointVector.push( *aStartIterator );

    while( aTime <  aLastTime )
      {
	const_iterator 
	  anIterator( theDataPointVector.lower_bound( aStartIterator,
						      anEndIterator,
						      aTime + anInterval ) );
	DataPointCref anDataPoint( *anIterator );
	aDataPointVector.push( anDataPoint );
	aTime = anDataPoint.getTime();
	aStartIterator = anIterator;
      }
    
    return aDataPointVector;
  }
  
  
  void Logger::appendData( RealCref aValue )
  {
    const Real aTime( (theGetCurrentTimeMethod)() );
    if( !theDataPointVector.empty() )
      {
    	theCurrentInterval = aTime - theDataPointVector.back().getTime();
      }
    theDataPointVector.push( aTime, aValue );
    if(theMinimumInterval < theCurrentInterval )
      {
	theMinimumInterval = theCurrentInterval; 
      }
  }

  //

  StringCref Logger::getName() const
    {
      return thePropertySlot->getName();
    }

  
  //
  
  RealCref Logger::getStartTime( void ) const
  {
    if(!theDataPointVector.empty())
      {
	return theDataPointVector.front().getTime();
      }
    else
      {
	static const Real aZero( 0.0 );
	return aZero;
      }
  }
  
  
  //
  
  RealCref Logger::getEndTime( void ) const
  {
    if(!theDataPointVector.empty())
      {
	return theDataPointVector.back().getTime();
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
