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
// 14/04/2002
//
// modified by Koichi Takahashi <shafi@e-cell.org>

#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include <functional>
#include <algorithm>
#include <boost/range/iterator_range.hpp>
#include "libecs.hpp"
#include "Polymorph.hpp"
#include "Logger.hpp"
#include "VVector.hpp"

namespace libecs
{

class Logger::LoggerImpl
{
public:
    typedef VVector<DataPoint> DataPointVector;

public:
    LoggerImpl( const LoggingPolicy& pol = LoggingPolicy() );
    virtual ~LoggerImpl();

    void push_back( const Logger::DataPoint& aDataPoint );

    void setPolicy( const LoggingPolicy& pol );

    const LoggingPolicy& getPolicy() const;

    Step size() const;

    const Logger::DataPoint& get( Logger::Step step ) const;

    Logger::DataPoint& get( Logger::Step step );

    Logger::DataPoint& operator[]( Logger::Step step )
    {
        return get( step );
    }

    const Logger::DataPoint& operator[]( Logger::Step step ) const
    {
        return get( step );
    }

private:
    Time             lastTime_;
    Step             stepCount_;
    DataPointVector* vec_;
    LoggingPolicy    policy_;
};

Logger::LoggerImpl::LoggerImpl( const LoggingPolicy& pol )
        : policy_( pol ),
          vec_( VVectorMaker::getInstance().create<DataPoint>() ),
          lastTime_( 0.0 ), stepCount_( 0 )
{
}

Logger::LoggerImpl::~LoggerImpl()
{
    delete vec_;
}

void Logger::LoggerImpl::setPolicy( const LoggingPolicy& policy )
{
    policy_ = policy;
}

const LoggingPolicy& Logger::LoggerImpl::getPolicy() const
{
    return policy_;
}
    
Logger::DataPoint& Logger::LoggerImpl::get( Logger::Step index )
{
    return vec_->at( index );
}

const Logger::DataPoint& Logger::LoggerImpl::get( Logger::Step index ) const
{
    return vec_->at( index );
}

void Logger::LoggerImpl::push_back( const DataPoint& dp )
{
    ++stepCount_;
    if ( policy_.getMinimumInterval() == 0.0 )
    {
        if ( stepCount_ < policy_.getMinimumStep() )
        {
            return;
        }
    }
    else
    {
        if ( ( dp.time - lastTime_ ) < policy_.getMinimumInterval() )
        {
            return;
        }
    }

    vec_->push_back( dp );

    lastTime_ = dp.time;
    stepCount_ = 0;
}

Logger::Step Logger::LoggerImpl::size() const
{
    return vec_->size();
}

Logger::iterator::reference
Logger::iterator::operator*() const
{
    return impl_->get( idx_ );
}

Logger::iterator::pointer
Logger::iterator::operator->() const
{
    return &impl_->get( idx_ );
}

Logger::iterator::reference
Logger::iterator::operator[]( difference_type idx ) const
{
    return impl_->get( idx_ + idx );
}

Logger::const_iterator::reference
Logger::const_iterator::operator*() const
{
    return impl_->get( idx_ );
}

Logger::const_iterator::pointer
Logger::const_iterator::operator->() const
{
    return &impl_->get( idx_ );
}

Logger::const_iterator::reference
Logger::const_iterator::operator[]( difference_type idx ) const
{
    return impl_->get( idx_ + idx );
}


Logger::Logger( const LoggingPolicy& pol )
        : impl_( new LoggerImpl() )
{
}

Logger::~Logger()
{
}

void Logger::setPolicy( const LoggingPolicy& pol )
{
    impl_->setPolicy( pol );
}

const LoggingPolicy& Logger::getPolicy() const
{
    return impl_->getPolicy();
}

void Logger::push_back( const DataPoint& dp )
{
    impl_->push_back( dp );
}

Logger::Step Logger::size() const
{
        return impl_->size();
}

const Logger::DataPoint& Logger::front() const
{
    if ( size() == 0 )
    {
        THROW_EXCEPTION( IllegalOperation, "no data available" );
    }

    return impl_->get(0);
}

const Logger::DataPoint& Logger::back() const
{
    if ( size() == 0 )
    {
        THROW_EXCEPTION( IllegalOperation, "no data available" );
    }

    return impl_->get(impl_->size() - 1);
}

Logger::iterator Logger::begin()
{
    return iterator( impl_, 0 );
}

Logger::const_iterator Logger::begin() const
{
    return const_iterator( impl_, 0 );
}

Logger::iterator Logger::end()
{
    return iterator( impl_, size() );
}

Logger::const_iterator Logger::end() const
{
    return const_iterator( impl_, size() );
}

Logger::iterator Logger::find( Time time )
{
    if ( size() == 0 )
    {
        THROW_EXCEPTION( IllegalOperation, "no data available" );
    }

    Time avgIntervalPerStep(
            ( back().time - front().time ) / size() );

    Step s( static_cast< Step >( time / avgIntervalPerStep ) );

    if ( impl_->get( s ).time <= time )
    {
        Logger::iterator i( std::find_if( begin() + s, end(),
                std::not1( DataPoint::Earliness( time ) ) ) );
        return i;
    }
    else if ( impl_->get( s ).time > time )
    {
        Logger::iterator i( std::find_if( begin() + s, begin(),
                DataPoint::Earliness( time ) ) );
        if ( i->time < time )
        {
            ++i;
        }
        return i;
    }
    return begin() + s;
}

Logger::const_iterator Logger::find( Time time ) const
{
    if ( size() == 0 )
    {
        THROW_EXCEPTION( IllegalOperation, "no data available" );
    }

    Time avgIntervalPerStep(
            ( back().time - front().time ) / size() );

    Step s( static_cast< Step >( time / avgIntervalPerStep ) );

    if ( impl_->get( s ).time <= time )
    {
        Logger::const_iterator i( std::find_if( begin() + s, end(),
                std::not1( DataPoint::Earliness( time ) ) ) );
        return i;
    }
    else if ( impl_->get( s ).time > time )
    {
        Logger::const_iterator i( std::find_if( begin() + s, begin(),
                DataPoint::Earliness( time ) ) );
        if ( i->time < time )
            ++i;
        return i;
    }
    return begin() + s;
}

Logger::DataPoints Logger::getDataPoints(Step startIdx, Step endIdx )
{
    DataPoints::iterator startPos( begin() + startIdx );
    DataPoints::iterator endPos(
            endIdx == static_cast<Logger::Step>( -1 ) ?
                end(): begin() + endIdx );
    if ( startPos >= end() || startPos < begin())
    {
        THROW_EXCEPTION(OutOfRange, "start index out of range: " + startIdx );
    }
    if ( endPos >= end() || endPos < begin())
    {
        THROW_EXCEPTION(OutOfRange, "start index out of range: " + endIdx );
    }
    return DataPoints( startPos, endPos );
}

Logger::ConstDataPoints Logger::getDataPoints(Step startIdx, Step endIdx ) const
{
    ConstDataPoints::iterator startPos( begin() + startIdx );
    ConstDataPoints::iterator endPos(
            endIdx == static_cast<Logger::Step>( -1 ) ?
                end(): begin() + endIdx );
    if ( startPos > end() || startPos < begin())
    {
        THROW_EXCEPTION(OutOfRange, "start index out of range: " + startIdx );
    }
    if ( endPos > end() || endPos < begin())
    {
        THROW_EXCEPTION(OutOfRange, "start index out of range: " + endIdx );
    }
    return ConstDataPoints( startPos, endPos );
}

Logger::DataPoints Logger::getDataPoints( Time startTime, Time endTime )
{
    return DataPoints( find( startTime ), find( endTime ) );
}

Logger::ConstDataPoints Logger::getDataPoints( Time startTime, Time endTime ) const
{
    return ConstDataPoints( find( startTime ), find( endTime ) );
}


} // namespace libecs

