/*

 */

#if !defined(__LOGGER_HPP)
#include "logger.hpp"
#endif


// Constructor


template <class T, class V>
Logger<T, V>::Logger(ObjectPtr optr)
  :
  theDataPointVector(*(new DataPointVector())),
  theDataFuncCptr(optr->fptr)
{
  ; // do nothing
} 

// Copy Constructor

template <class T, class V>
Logger<T, V>::Logger(const Logger& lg)
  :
  theDataPointVector(lg.getDataPointVector()),
  theDataFuncCptr(lg.getDataFuncCptr())
{
  ; // do nothing
}


// Destructor

template <class T, class V>
Logger<T,V>::~Logger(void)
{
  ;
  //  delete theDataPointVectorPtr;
}

//


template <class T, class V>
const Logger<T,V>::DataPointVector& Logger<T,V>::getData( const T& first ,
							  const T& last ,
							  const T& interval ) const
{
  
  DataPointVector* aDataPointVectorPtr = new DataPointVector();
  
  const_iterator itr_1 = theDataPointVector.begin();
  const_iterator itr_2 = theDataPointVector.end();
  
  const_iterator firstItr = binary_search(itr_1, itr_2, first);
  const_iterator lastItr = binary_search(itr_1, itr_2, last);

  const_iterator i = firstItr;
  T time = first;
  while(time < (*lastItr)->getTime())
    {
      const_iterator n = 
	theDataPointVector.binary_search( i,
					  lastItr,
					  time+interval );
      aDataPointVectorPtr->push(**n);
      time = (*n)->getTime();
      i = n;
    }

  
  return *aDataPointVectorPtr;
}


//

template <class T, class V>
void Logger<T, V>::update( void )
{
  containee_type* cptr = new containee_type((*theDataFuncCptr)(), (*theDataFuncCptr)());
  appendData(*cptr);
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
#include "datapoint.cpp"
#include "stl_datapointvector.cpp"
typedef double Float;

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
