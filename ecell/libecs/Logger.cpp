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


#include "PropertyInterface.hpp"

/*

 */

#if !defined(__LOGGER_HPP)
#include "Logger.hpp"
#endif



namespace libecs
{

  // Constructor

  Logger::Logger( AbstractPropertySlotCref aPropertySlot )
    :
    thePropertySlot( aPropertySlot ),
    theMinimumInterval( 0.0 ),
    theCurrentInterval( 0.0 )
  {
    ; // do nothing
  } 
  
  // Destructor
  
  Logger::~Logger( void )
  {
    delete &thePropertySlot;
  }
  
  
  
  Logger::DataPointVectorCref Logger::getData( void ) const
  {
    const_iterator aItr = theDataPointVector.begin();
    const_iterator endItr = theDataPointVector.end();
    DataPointVectorPtr aDataPointVectorPtr( new DataPointVector() );
    
    while( aItr != endItr )
      {
	aDataPointVectorPtr->push( **aItr );
	aItr++;
      }
    
    return *aDataPointVectorPtr;
    
  }
  
  //
  
  Logger::DataPointVectorCref Logger::getData( RealCref aStartTime,
					       RealCref anEndTime ) const
  {
    const_iterator itr_1( theDataPointVector.begin() );
    const_iterator itr_2( theDataPointVector.end() );
    
    
    const_iterator 
      aStartIterator( theDataPointVector.binary_search( itr_1,
						       itr_2,
						       aStartTime ) );
    const_iterator 
      anEndIterator( theDataPointVector.binary_search( itr_1,
						     itr_2,
						     anEndTime ) );
    DataPointVectorPtr aNewDataPointVectorPtr( new DataPointVector() );
    while( aStartIterator != anEndIterator )
      {
	aNewDataPointVectorPtr->push( **aStartIterator );
	aStartIterator++;
      }
    
    return *aNewDataPointVectorPtr;
  }
  
  
  //
  
  
  Logger::DataPointVectorCref Logger::getData( RealCref aStartTime,
					       RealCref anEndTime,
					       RealCref anInterval ) const
  {
    
    DataPointVectorPtr aDataPointVectorPtr( new DataPointVector() );
    
    const_iterator aFirstIterator( binary_search( theDataPointVector.begin(),
						  theDataPointVector.end(),
						  aStartTime ) );

    const_iterator aLastIterator( binary_search( aFirstIterator,
						 theDataPointVector.end(),
						 anEndTime ) );

    
    Real aTime( aStartTime );
    Real aLastTime( (*aLastIterator)->getTime() );
    while( aTime <  aLastTime ) // FIXME
      {
	const_iterator 
	  anIterator( theDataPointVector.binary_search( aFirstIterator,
							aLastIterator,
							aTime + anInterval ) );
	aDataPointVectorPtr->push( **anIterator );
	aTime = (*anIterator)->getTime();
	aFirstIterator = anIterator;
      }
    
    return *aDataPointVectorPtr;
  }
  
  
  //
  
  void Logger::appendData( const containee_type& aDataPoint )
  {
    theCurrentInterval = 
      aDataPoint.getTime() - theDataPointVector.back()->getTime();
    theDataPointVector.push( aDataPoint );
    if( theMinimumInterval < theCurrentInterval )
      {
	theMinimumInterval = theCurrentInterval; 
      }
  }
  
  
  //
  
  void Logger::appendData( RealCref aTime, UConstantCref aValue )
  {
    theCurrentInterval = aTime - theDataPointVector.back()->getTime();
    theDataPointVector.push( aTime, aValue );
    if(theMinimumInterval < theCurrentInterval )
      {
	theMinimumInterval = theCurrentInterval; 
      }
  }
  
  
  //
  
  RealCref Logger::getStartTime( void ) const
  {
    return theDataPointVector[0]->getTime();
  }
  
  
  //
  
  RealCref Logger::getEndTime( void ) const
  {
    return theDataPointVector.back()->getTime();
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

  Real f1 = 0.12;
  Real f2 = 1.23;
  Real f3 = 3.14;
  Real f4 = 4.27;
  Real f5 = 7.23;
  
  DataPoint<Real,Real> d1 = DataPoint<Real,Real>(f1,f1);
  DataPoint<Real,Real> d2 = DataPoint<Real,Real>(f2,f2);
  DataPoint<Real,Real> d3 = DataPoint<Real,Real>(f3,f3);
  DataPoint<Real,Real> d4 = DataPoint<Real,Real>(f4,f4);
  DataPoint<Real,Real> d5 = DataPoint<Real,Real>(f5,f5);

  ObjectPtr op = new Object(&func);


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
