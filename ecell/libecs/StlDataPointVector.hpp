#if !defined(__STL_DATAPOINTVECTOR_HPP)
#define __STL_DATAPOINTVECTOR

#include <vector>

#include "libecs.hpp"

#include "DataPoint.hpp"


/**

 */


template<class T, class V, class Containee = DataPoint<T,V>,
	 class Container = vector<Containee*> >
class StlDataPointVector
{

public:
  typedef Containee containee_type;
  typedef typename Container::value_type value_type;
  typedef typename Container::size_type size_type;
  typedef typename Container::iterator iterator;
  typedef typename Container::const_iterator const_iterator;
  typedef typename Container::reference reference;
  typedef typename Container::const_reference const_reference;


public:

  explicit StlDataPointVector()
  {
    ; // do nothing
  }

  explicit StlDataPointVector( const Container& vect )
    :
    theContainer( vect )
  {
    ; // do nothing
  }

  StlDataPointVector( const StlDataPointVector& );

  StlDataPointVector( size_type sz )
  {
    theContainer = Container( sz );
  }

  ~StlDataPointVector(void);

  reference operator[] ( size_type sz )
  {
    return *(theContainer[sz]);
  }


  const_reference operator[] ( size_type sz ) const
  {
    return *(theContainer[sz]);
  }

  bool empty() const
  {
    return theContainer.empty();
  }

  size_type size() const
  {
    return theContainer.size();
  }

  const_iterator begin() const
  {
    return theContainer.begin();
  }

  iterator begin()
  {
    return theContainer.begin();
  }

  const_iterator end() const
  {
    return theContainer.end();
  }

  iterator end()
  {
    return theContainer.end();
  }

  void push( const containee_type& x );

  void push( const T&, const V& );

  const_iterator binary_search( const_iterator first, const_iterator last,
				const T&) const;

  /*  const_iterator binary_search(size_type, size_type, const T&) const; */

private:

  Container theContainer;

};


#endif /* __DATAPOINTVECTOR_HPP */
