#if !defined(__LOGGER_HPP)
#define __LOGGER_HPP


/*

 */

#include "datapoint.hpp"

template <class T, class V> class DataPoint;
template <class T, class V> class Logger;

typedef double Float;                          // FIXME :temp
typedef const Float& (*const DataFuncCptr)( ); // FIXME:temp 

typedef double Time;      // FIXME : temporary use
typedef double ValueType; // FIXME : temporary use 

typedef const Float& (*const funcptr)();

class Object
{
public:
  Object(funcptr fp)
    :
    fptr(fp)
  {
    ;
  }

  funcptr fptr;
};

typedef Object* ObjectPtr; // FIXME : temp 


#if defined(STL_DATAPOINTVECTOR)
#include "stl_datapointvector.hpp"
#endif /* END OF STL_DATAPOINTVECTOR */



/**
   
 */

template <class T, class V>
class Logger
{
public:
#if defined(STL_DATAPOINTVECTOR)
  typedef StlDataPointVector<T,V> DataPointVector;
#endif /* END OF STL_DATAPOINTVECTOR */

#if defined(VVECTOR)
  typedef VVector DataPointVector;
#endif /* END OF VVECTOR */ 


  typedef typename DataPointVector::containee_type containee_type;
  typedef typename DataPointVector::const_iterator const_iterator;
  typedef typename DataPointVector::iterator iterator;
  typedef typename DataPointVector::size_type size_type;

  
public:

  /**
     Constructor
  */
  
  Logger( ObjectPtr );
  
  /**
     Copy constructor
  */
  
  Logger( const Logger& );


  /// Destructor

  ~Logger( void );


  /**

   */

  const DataPointVector& getData( const T&, const T&, const T& ) const;


  void update( void );
  
  //FIXME temp
  void update( containee_type& dp )
  {
    appendData(dp);
  }


  void push(const containee_type& x)
  {
    theDataPointVector.push(x);
  }

  void push(const T& t, const V& v)
  {
    theDataPointVector.push(t,v);
  }

  const_iterator binary_search(const_iterator it1, const_iterator it2, const T& t) const
  {
    return theDataPointVector.binary_search(it1,it2,t);
  }

  const_iterator binary_search(size_type s1, size_type s2, const T& t) const
  {
    return theDataPointVector.binary_search(s1,s2,t);
  }





  //
  // Protected and Private methods follow
  //

protected:
  
  
  /**

   */

  void appendData( const containee_type& );


private:


  /// Default constructor is hidden
  
  Logger( void );

  /// Assignment operator is hidden
  
  Logger<T, V>& operator=( const Logger<T,V>& );
  

  /**

   */
  const DataPointVector& getDataPointVector( void ) const
  {
    return theDataPointVector;
  }

  /**

   */
  const DataFuncCptr& getDataFuncCptr( void ) const
  {
    return theDataFuncCptr;
  }


private:
  

  //
  // Protected and Private data members follow
  //


private:

  /// Data members

  DataPointVector theDataPointVector;
  DataFuncCptr theDataFuncCptr;


};


#endif /* __LOGGER_HPP */
