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



/*

 */

#if !defined(__LOGGER_HPP)
#include "Logger.hpp"
#endif

namespace libecs
{

  // Default Constructor

  Logger::Logger( void )
    :
    theMinimumInterval( 0.0 ),
    theCurrentInterval( 0.0 )
  {
    theMessageSlotClassCptr = 0;
  } 

  // Constructor


  Logger::Logger( AbstractMessageSlotClassCptr aMessageSlotClassPtr )
    :
    theMessageSlotClassCptr( aMessageSlotClassPtr ),
    theMinimumInterval( 0.0 ),
    theCurrentInterval( 0.0 )
  {
    ; // do nothing
  } 
  
  // Copy Constructor
  
  Logger::Logger( LoggerCref logger )
    :
    theDataPointVector( logger.getDataPointVector() ),
    theMessageSlotClassCptr( logger.getMessageSlotClassCptr() ),
    theMinimumInterval( logger.getMinInterval() ),
    theCurrentInterval( logger.getCurrentInterval() )
  {
    ; // do nothing
  }
  
  
  // Destructor
  
  Logger::~Logger( void )
  {
    ; // do nothing
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
  
  Logger::DataPointVectorCref Logger::getData( RealCref start,
					       RealCref end ) const
  {
    const_iterator itr_1 = theDataPointVector.begin();
    const_iterator itr_2 = theDataPointVector.end();
    
    
    const_iterator startItr = theDataPointVector.binary_search( itr_1,
								itr_2,
								start );
    const_iterator endItr = theDataPointVector.binary_search( itr_1,
							      itr_2,
							      end );
    const_iterator i = startItr;
    DataPointVectorPtr aNewDataPointVectorPtr( new DataPointVector() );
    while( i != endItr )
      {
	aNewDataPointVectorPtr->push( **i );
	i++;
      }
    
    return *aNewDataPointVectorPtr;
  }
  
  
  //
  
  
  Logger::DataPointVectorCref Logger::getData( RealCref first,
					       RealCref last,
					       RealCref interval ) const
  {
    
    DataPointVectorPtr aDataPointVectorPtr( new DataPointVector() );
    
    const_iterator itr_1 = theDataPointVector.begin();
    const_iterator itr_2 = theDataPointVector.end();
    
    const_iterator firstItr = binary_search( itr_1, itr_2, first );
    const_iterator lastItr  = binary_search( itr_1, itr_2, last );
    
    const_iterator i = firstItr;
    Real aTime( first );
    while( aTime < (*lastItr)->getTime() ) // FIXME
      {
	const_iterator n = 
	  theDataPointVector.binary_search( i,
					    lastItr,
					    aTime + interval
					    );
	aDataPointVectorPtr->push( **n );
	aTime = (*n)->getTime();
	i = n;
      }
    
    return *aDataPointVectorPtr;
  }
  
  
  //
  // FIXME
  /*
  void Logger::update( void )
  {
    appendData( containee_type( (*theDataFuncCptr)(), (*theDataFuncCptr)() ) );
  }
  */
  
  //
  
  void Logger::appendData(const containee_type& dp )
  {
    theCurrentInterval = dp.getTime() - theDataPointVector.back()->getTime();
    theDataPointVector.push( dp );
    if(theMinimumInterval < theCurrentInterval )
      {
	theMinimumInterval = theCurrentInterval; 
      }
  }
  
  
  //
  
  void Logger::appendData( RealCref t, UniversalVariableCref v )
  {
    theCurrentInterval = t - theDataPointVector.back()->getTime();
    theDataPointVector.push( t, v );
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
  lg.update(d1);
  lg.update(d2);
  lg.update(d3);
  lg.update(d4);
  lg.update(d5);


  Logger<Real,Real> lg_clone = Logger<Real,Real>(lg);

  //  printf("%p %p\n",&(lg.getDataPointVector()),&(lg_clone.getDataPointVector()));

  lg.getData(0.0,5.0,0.5);
  printf("%f\n",lg.getStartTime());
  printf("%f\n",lg.getEndTime());
}



#endif /* LOGGER_TEST */
