//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2008 Keio University
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
// written by Masayuki Okayama <smash@e-cell.org>,
// E-Cell Project.
//
// modified by Gabor Bereczki <gabor.bereczki@talk21.com>
// modified by Koichi Takahashi <shafi@e-cell.org>

#if !defined(__LOGGER_HPP)
#define __LOGGER_HPP

#include <vector>
#include <cstddef>

#include <boost/utility.hpp>

#include "libecs.hpp"
#include "LoggingPolicy.hpp"
#include "DataPoint.hpp"

/*
 // enumeration for logging policy
 enum Policy
   {
STEP_SIZE = 0,
TIME_INTERVAL,
END_POLICY,
MAX_SPACE
   };
*/

/**
   @addtogroup logging The Data Logging Module.
   The Data Logging Module.

   @ingroup libecs
 */
/** @{ */

/** @file */
namespace libecs
{

/**
 Logger module for logging and retrieving data runtime.
 */
class LIBECS_API Logger
{
private:
    class LoggerImpl;

public:
    typedef DataPoint< Time, Real > DataPoint;
    typedef ::std::size_t Step;
    typedef ::std::ptrdiff_t StepDifference;

    typedef ::std::size_t size_type;
    typedef ::std::ptrdiff_t difference_type;
    typedef DataPoint value_type;
    typedef DataPoint* pointer;
    typedef const DataPoint* const_pointer;
    typedef DataPoint& reference;
    typedef const DataPoint& const_reference;

    template< typename Timpl_, typename Tself_ >
    class iterator_base
    {
    protected:
        typedef boost::shared_ptr< Timpl_ > ImplHandle;
        typedef Tself_ Self;

    public:
        typedef Logger::value_type value_type;
        typedef Logger::difference_type difference_type;
        typedef ::std::bidirectional_iterator_tag iterator_category;

    public:
        iterator_base( ImplHandle impl, Step idx )
            : impl_( impl ), idx_( idx ) {}

        Self& operator--()
        {
            --idx_;
            return *static_cast<Self*>(this);
        }

        Self operator--(int)
        {
            Self retval_( *this );
            --idx_;
            return retval_;
        }

        Self& operator++()
        {
            ++idx_;
            return *static_cast<Self*>(this);
        }

        Self operator++(int)
        {
            Self retval_( *this );
            ++idx_;
            return retval_;
        }

        Self operator+(difference_type offset) const
        {
            return Self(impl_, idx_ + offset);
        }

        Self operator-(difference_type offset) const
        {
            return Self(impl_, idx_ - offset);
        }

        Self& operator+=(difference_type offset)
        {
            idx_ += offset;
            return *this;
        }

        Self& operator-=(difference_type offset)
        {
            idx_ += offset;
            return *this;
        }

        bool operator<( const Self& rhs ) const
        {
            return idx_ < rhs.idx_;
        }

        bool operator>( const Self& rhs ) const
        {
            return idx_ > rhs.idx_;
        }

        bool operator==( const Self& rhs ) const
        {
            return idx_ == rhs.idx_;
        }

        bool operator!=( const Self& rhs ) const
        {
            return !operator==( rhs );
        }

        bool operator>=( const Self& rhs ) const
        {
            return !operator<( rhs );
        }

        bool operator<=( const Self& rhs ) const
        {
            return !operator>( rhs );
        }

    protected:
        Step idx_;
        ImplHandle impl_;
    };

    class iterator
            : public iterator_base< LoggerImpl, iterator >
    {
    protected:
        typedef iterator_base< LoggerImpl, iterator > Base;

    public:
        typedef Logger::pointer pointer;
        typedef Logger::reference reference;
        typedef Base::iterator_category iterator_category;
 
    public:
        iterator( Base::ImplHandle impl, Step idx )
            : Base( impl, idx ) {}

        iterator( const Base& that )
            : Base( that ) {}

        reference operator*() const;

        pointer operator->() const;

        reference operator[]( difference_type idx ) const;
    };

    class const_iterator
            : public iterator_base< const LoggerImpl, const_iterator >
    {
    protected:
        typedef iterator_base< const LoggerImpl, const_iterator > Base;

    public:
        typedef Logger::const_pointer pointer;
        typedef Logger::const_reference reference;
        typedef Base::iterator_category iterator_category;
 
    public:
        const_iterator( Base::ImplHandle impl, Step idx )
            : Base( impl, idx ) {}

        const_iterator( const Base& that )
            : Base( that ) {}

        reference operator*() const;

        pointer operator->() const;

        reference operator[]( difference_type idx ) const;
    };

    typedef boost::iterator_range<iterator> DataPoints;
    typedef boost::iterator_range<const_iterator> ConstDataPoints;

public:
    Logger( const LoggingPolicy& pol = LoggingPolicy() );

    ~Logger();

    /**
      Sets logging policy
     */
    void setPolicy( const LoggingPolicy& pol );

    /**
      Returns logging policy vector.
    */
    const LoggingPolicy& getPolicy( void ) const;

    /**
      Log current value that theLoggerAdapter gives with aTime.
    */
    void log( const DataPoint& dp );

    /**
       Returns time of the first element  in Logger.
    */
    const Time getStartTime() const;

    /**
       Returns time of the last element in Logger
    */
    const Time getEndTime() const;

    const TimeDifference getAverageInterval() const
    {
        return size() == 0.0 ? 0.0:
            ( getEndTime() - getStartTime() ) / size();
    }

    /**
      Returns size of logger
    */
    size_type size() const;

    iterator begin();

    const_iterator begin() const;

    iterator end();

    const_iterator end() const;

    iterator find( Time time );

    const_iterator find( Time time ) const;

    DataPoints getDataPoints(Step startIdx = 0, Step endIdx = static_cast<Step>( -1 ));

    ConstDataPoints getDataPoints(Step startIdx = 0, Step endIdx = static_cast<Step>( -1 )) const;

    DataPoints getDataPoints( Time startTime, Time endTime = -1.0 );

    ConstDataPoints getDataPoints( Time startTime, Time endTime = 1.0) const;

private:
    /// Data members
    boost::shared_ptr< LoggerImpl > impl_;
};

} // namespace libecs

/** @} */ // logging module


#endif /* __LOGGER_HPP */

