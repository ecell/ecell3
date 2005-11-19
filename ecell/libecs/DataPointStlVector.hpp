//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2000-2002 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Masayuki Okayama <smash@e-cell.org>,
// E-Cell Project.
//


#if !defined(__DATAPOINT_STL_VECTOR_HPP)
#define __DATAPOINT_STL_VECTOR_HPP

#include <vector>
#include <algorithm>

#include "libecs.hpp"

#include "DataPoint.hpp"

namespace libecs
{

  /** @addtogroup logging
   *@{
   */

  /** @file */


  DECLARE_CLASS(DataPointStlVector);
  DECLARE_TYPE(DataPoint,Containee);
  DECLARE_VECTOR(Containee,Container);


  class DataPointStlVector
  {

  public:

    typedef Container::const_iterator  const_iterator;
    typedef Container::iterator        iterator;
    typedef Container::const_reference const_reference;
    typedef Container::reference       reference;
    typedef Container::size_type       size_type;
    
    explicit DataPointStlVector( void )
    {
      ; // do nothing
    }

    DataPointStlVector( const_iterator start, const_iterator end )
      :
      theContainer( start, end )
    {
      ; // do nothing
    }


    DataPointStlVector( DataPointStlVectorCref vector )
      :
      theContainer( vector.getContainer() )
    {
      ; // do nothing
    }

    DataPointStlVector( DataPointStlVectorRef vector )
      :
      theContainer( vector.getContainer() )
    {
      ; // do nothing
    }


    ~DataPointStlVector(void);

    ContainerCref getContainer() const
    {
      return theContainer;
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

    const_iterator end() const
    {
      return theContainer.end();
    }

    const_reference front() const
    {
      return theContainer.front();
    }

    const_reference back() const
    {
      return theContainer.back();
    }

    iterator begin()
    {
      return theContainer.begin();
    }

    iterator end()
    {
      return theContainer.end();
    }

    reference back()
    {
      return theContainer.back();
    }


    void push( ContaineeRef aRef )
    {
      theContainer.push_back( aRef );
    }

    void push( ContaineeCref aCref )
    {
      theContainer.push_back( aCref );
    }

    void push( ContaineePtr aPtr )
    {
      theContainer.push_back( *aPtr );
    }


    void push( RealParam t, PolymorphCref v )
    {
      theContainer.push_back( Containee( t, v ) );
    }

    void push( RealParam t, RealParam v )
    {
      theContainer.push_back( Containee( t, v ) );
    }

    static const_iterator lower_bound( const_iterator first,
				       const_iterator last,
				       RealParam aTime ) 
    {
      return std::lower_bound( first, last, aTime );
    }

    static const_iterator upper_bound( const_iterator first,
				       const_iterator last,
				       RealParam aTime ) 
    {
      return std::upper_bound( first, last, aTime );
    }


  private:
    Container theContainer;


  };

  //@}

} // namespace libecs

#endif /* __DATAPOINT_STL_VECTOR_HPP */
