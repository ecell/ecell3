/*





 */

#if !defined(__DATAPOINT_HPP)
#include "DataPoint.hpp"
#endif



// Constructor that sets the
// internal value

template <class T, class V>
DataPoint<T,V>::DataPoint( const T& t, const V& v )
  :
  theTime( t ),
  theValue( v )
 { 
   ; // do nothing
 }


// Copy constructor 

template <class T, class V>
DataPoint<T,V>::DataPoint( DataPointCref datapoint )
  :
  theTime( datapoint.getTime() ),
  theValue( datapoint.getValue() )
{
  ; // do nothing
}



// Assignment operator from another DataPoint

template <class T, class V>
DataPoint<T,V>& DataPoint<T,V>::operator=( DataPointCref rhs )
{
  if( this == &rhs )
    {
      return *this;
    }

  theTime  = rhs.getTime();
  theValue = rhs.getValue();

  return *this;
}


#if defined(DATAPOINT_TEST)
#include <stdio.h>

main()
{
  DataPoint<double,double> dp1 = DataPoint<double,double>(3.14,3.14);
  DataPoint<double,double> dp2 = DataPoint<double,double>(5.14,3.14);
  if(dp1 < dp2 )
    {
      printf("true\n");
      return 1;
    }
  printf("false\n");
  return 1;
  

}
#endif
