//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2009 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell System is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell System is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell System -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Gabor Bereczki <gabor.bereczki@talk21.com>
//

#if !defined(__PHYSICALLOGGER_HPP)
#define __PHYSICALLOGGER_HPP

#include "libecs/Defs.hpp"
#include "libecs/Exceptions.hpp"
#include "libecs/DataPoint.hpp"
#include "libecs/DataPointVector.hpp"

template <typename T> class vvector;

namespace libecs
{

class LIBECS_API PhysicalLogger
{
    typedef vvector< DataPoint > Vector;

public:
    DECLARE_TYPE( size_t, VectorIterator );
    DECLARE_TYPE( size_t, size_type );

    PhysicalLogger();
    virtual ~PhysicalLogger();
            
    void push( DataPointCref aDataPoint );

    void setEndPolicy( Integer anEndPolicy );

    int getEndPolicy() const;

    /// set max storage size in Kbytes.
    void setMaxSize( size_type aMaxSize );

    size_type getMaxSize() const
    {
        return theMaxSize;
    }

    size_type lower_bound( const size_type start,
                           const size_type end,
                           const Real time ) const;

    size_type upper_bound( const size_type start,
                           const size_type end,
                           const Real time ) const;

    size_type lower_bound_linear( const size_type start,
                                  const size_type end,
                                  const Real time ) const;

    size_type upper_bound_linear( const size_type start,
                                  const size_type end,
                                  const Real time ) const;

    size_type lower_bound_linear_backwards( const size_type start,
                                            const size_type end,
                                            const Real time ) const;

    size_type lower_bound_linear_estimate( const size_type start,
                                           const size_type end,
                                           const Real time,
                                           const Real time_per_step ) const;

    size_type upper_bound_linear_estimate( const size_type start,
                                           const size_type end,
                                           const Real time,
                                           const Real time_per_step ) const;
    
    boost::shared_ptr< DataPointVector > getVector( const size_type start,
                                                    const size_type end ) const;

    size_type size() const;

    bool empty() const;


    LongDataPoint at( size_type index) const;

    size_type begin() const
    {
        return 0;
    }
    
    
    size_type end() const
    {
        if ( size() > 0 )
        {
            return size() - 1;
        }
        else
        {
            return 0;
        }
    }


    LongDataPoint front() const
    {
        if ( empty() )
        {
            return DataPoint();
        }

        return at( begin() );
    }
    
    LongDataPoint back() const
    {
        if ( empty() )
        {
            return DataPoint();
        }

        return at( end() );
    }

    Real getAverageInterval() const;

private:

    // this mutable can be removed if vvector supports const operations
    Vector                 *theVector;

    size_type            theMaxSize;

};

} // namespace libecs

#endif /* __PHYSICALLOGGER_HPP */
