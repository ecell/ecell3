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


// Constructor


template <class T, class V>
Logger<T, V>::Logger( ObjectPtr optr )
  :
  theDataFuncCptr( optr->fptr )
{
  ; // do nothing
} 

// Copy Constructor

template <class T, class V>
Logger<T, V>::Logger( LoggerCref logger )
  :
  theDataPointVector( logger.getDataPointVector() ),
  theDataFuncCptr( logger.getDataFuncCptr() )
{
  ; // do nothing
}


// Destructor

template <class T, class V>
Logger<T,V>::~Logger( void )
{
  ; // do nothing
}

//


template <class T, class V>
const Logger<T,V>::DataPointVector&
Logger<T,V>::getData( const T& first,
		      const T& last,
		      const T& interval ) const
{
  
  DataPointVector* aDataPointVectorPtr( new DataPointVector() );
  
  const_iterator itr_1 = theDataPointVector.begin();
  const_iterator itr_2 = theDataPointVector.end();
  
  const_iterator firstItr = binary_search( itr_1, itr_2, first );
  const_iterator lastItr  = binary_search( itr_1, itr_2, last );

  const_iterator i = firstItr;
  T aTime( first );
  while( aTime < (*lastItr)->getTime())
    {
      const_iterator n = 
	theDataPointVector.binary_search( i,
					  lastItr,
					  aTime + interval );
      aDataPointVectorPtr->push( **n );
      aTime = (*n)->getTime();
      i = n;
    }

  return *aDataPointVectorPtr;
}


//

template <class T, class V>
void Logger<T, V>::update( void )
{
  appendData( containee_type( (*theDataFuncCptr)(), (*theDataFuncCptr)() ) );
}


//

template <class T, class V>
void Logger<T, V>::appendData(const containee_type& dp )
{
  theDataPointVector.push( dp );
}




#ifdef LOGGER_TEST


#include <stdio.h>
#include <iostream>
#include "DataPoint.cpp"
#include "StlDataPointVector.cpp"

const Float& func(void)
{
  const Float* fp = new Float(3.14); 
  return *fp;
}

main()
{

  Float f1 = 0.12;
  Float f2 = 1.23;
  Float f3 = 3.14;
  Float f4 = 4.27;
  Float f5 = 7.23;
  
  DataPoint<Float,Float> d1 = DataPoint<Float,Float>(f1,f1);
  DataPoint<Float,Float> d2 = DataPoint<Float,Float>(f2,f2);
  DataPoint<Float,Float> d3 = DataPoint<Float,Float>(f3,f3);
  DataPoint<Float,Float> d4 = DataPoint<Float,Float>(f4,f4);
  DataPoint<Float,Float> d5 = DataPoint<Float,Float>(f5,f5);

  ObjectPtr op = new Object(&func);


  Logger<Float,Float> lg = Logger<Float,Float>(op);
  lg.update(d1);
  lg.update(d2);
  lg.update(d3);
  lg.update(d4);
  lg.update(d5);


  Logger<Float,Float> lg_clone = Logger<Float,Float>(lg);

  //  printf("%p %p\n",&(lg.getDataPointVector()),&(lg_clone.getDataPointVector()));

  lg.getData(0.0,5.0,0.5);

}

#endif /* LOGGER_TEST */
