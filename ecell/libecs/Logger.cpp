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

#include "libecs.hpp"
#include "Polymorph.hpp"

#include "Logger.hpp"
#include "PhysicalLogger.hpp"

namespace libecs
{

Logger::Logger( const LoggingPolicy& pol )
	: policy_( pol ), impl_( new PhysicalLogger() )
{
}

Logger::~Logger()
{
}

void Logger::setPolicy( const LoggingPolicy& pol )
{
    policy_ = pol;
    impl_.setPolicy( pol );
}

const LoggingPolicy& Logger::getPolicy()
{
    return policy_;
}

DataPointVectorSharedPtr Logger::getData() const
{
    return impl_.getVector( impl_.begin(), impl_.end() );
}

DataPointVectorSharedPtr Logger::getData( RealParam aStartTime,
        RealParam anEndTime ) const
{
    PhysicalLogger::size_type
        topIterator( impl_.upper_bound(
            impl_.begin(), impl_.end(), anEndTime ) ),
        bottomIterator( impl_.lower_bound(
            impl_.begin(), topIterator, aStartTime ) );

    return impl_.getVector( bottomIterator, topIterator );
}

void Logger::log( DataPoint dp )
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
        if ( ( aTime - lastTime_ ) < policy_.getMinimumInterval() )
        {
            return;
        }
    }

    impl_.push( dp );

    lastTime_ = aTime;
    stepCount_ = 0;
}

const Real Logger::getStartTime( void ) const
{
    return  impl_.front().getTime();

}

const Real Logger::getEndTime( void ) const
{
    return impl_.back().getTime();
}

Step Logger::getSize() const
{
	return impl_.size();
}

DataPointVectorSharedPtr Logger::getData( RealParam aStartTime,
        RealParam anEndTime,
        RealParam anInterval ) const
{
    // this case doesn't work well with below routine on x86-64.
    // anyway a serious effort of code cleanup is necessary.
    if ( aStartTime == anEndTime )
    {
        PhysicalLogger::size_type
        anIterator( impl_.upper_bound
                    ( impl_.begin(),
                      impl_.end(),
                      anEndTime ) );
        LongDataPoint aLongDataPoint( impl_.at( anIterator ) );
        DataPointVectorPtr aDataPointVector( new DataPointVector( 1, 5 ) );
        aDataPointVector->asLong( 0 ) = aLongDataPoint;
        return DataPointVectorSharedPtr( aDataPointVector );
    }

    // set up output vector
    DataPointVectorIterator aPhysicalRange( static_cast<size_type>(
            ( anEndTime - aStartTime ) / anInterval ) );

    //this is a technical adjustment, because I realized that sometimes
    //conversion from real is flawed: rounding error
    Real anEstimatedRange( ( anEndTime - aStartTime ) / anInterval );

    if ( ( static_cast<Real>( aPhysicalRange ) ) + 0.9999
            < anEstimatedRange )
    {
        ++aPhysicalRange;
    }

    Real aTimeGap( ( impl_.back().getTime()
                     - impl_.front().getTime() ) /
                   impl_.size() );

    DataPointVectorPtr
    aDataPointVector( new DataPointVector( aPhysicalRange, 5 ) );

    // set up iterators
    PhysicalLogger::size_type
    anIterationEnd( impl_.
                    upper_bound_linear_estimate
                    ( impl_.begin(),
                      impl_.end(),
                      anEndTime,
                      aTimeGap ) );

    PhysicalLogger::size_type
    anIterationStart( impl_.
                      lower_bound_linear_estimate
                      ( impl_.begin(),
                        anIterationEnd,
                        aStartTime,
                        aTimeGap ) );

    // start from vectorslice start to vectorslice end,
    // scan through all datapoints
    size_type aCounter( anIterationStart );

    Real aTargetTime( aStartTime + anInterval );
    LongDataPoint aLongDataPoint( impl_.at( aCounter ) );

    DataPointAggregator anAggregator;
    anAggregator.aggregate( aLongDataPoint );
    for ( DataPointVectorIterator anElementCount( 0 );
            anElementCount < aPhysicalRange; ++anElementCount )
    {
        do
        {
            if ( ( aCounter < anIterationEnd ) &&
                    ( aLongDataPoint.getTime() < aTargetTime ) )
            {
                ++aCounter;
                aLongDataPoint = impl_.at( aCounter );
            }

            anAggregator.aggregate( aLongDataPoint );
        } while ( aLongDataPoint.getTime() < aTargetTime &&
                aCounter < anIterationEnd );

        aDataPointVector->asLong( anElementCount ) = anAggregator.getData();
        anAggregator.beginNextPoint();

        aTargetTime += anInterval;
    }

    return DataPointVectorSharedPtr( aDataPointVector );
}

} // namespace libecs

