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


#if !defined(__STL_DATAPOINTVECTOR_HPP)
#define __STL_DATAPOINTVECTOR

#include <vector>

#include "libecs.hpp"

#include "DataPoint.hpp"

namespace libecs
{

  /**

   */


  class StlDataPointVector
  {

  public:
    typedef DataPoint Containee;
    typedef vector<Containee*> Container;
    typedef Container::value_type value_type;
    typedef Container::size_type size_type;
    typedef Container::iterator iterator;
    typedef Container::const_iterator const_iterator;
    typedef Container::reference reference;
    typedef Container::const_reference const_reference;


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
      return theContainer[sz];
    }


    const_reference operator[] ( size_type sz ) const
    {
      return theContainer[sz];
    }

    bool empty() const
    {
      return theContainer.empty();
    }

    reference back()
    {
      return theContainer.back();
    }

    const_reference back() const
    {
      return theContainer.back();
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

    void push( const Containee& );

    void push( RealCref, UniversalVariableCref );

    const_iterator binary_search( const_iterator first,
				  const_iterator last,
				  RealCref) const;


  private:

    Container theContainer;

  };


} // namespace libecs

#endif /* __DATAPOINTVECTOR_HPP */
