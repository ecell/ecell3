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

class PhysicalLogger;

/**
 Logger module for logging and retrieving data runtime.
 */
class LIBECS_API Logger
{
public:
    typedef DataPoint< Time, Real > DataPoint;
    typedef ::std::size_t Step;

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
    const LoggingPolicy& getPolicy( void );

    /**
      Log current value that theLoggerAdapter gives with aTime.
    */
    void log( const DataPoint& aTime );

    /**
       Returns time of the first element  in Logger.
    */
    const Real getStartTime( void ) const;

    /**
       Returns time of the last element in Logger
    */
    const Real getEndTime( void ) const;

    /**
      Returns size of logger
    */
    const Step getSize() const;

private:
    /// Data members
    boost::scoped_ptr< PhysicalLogger > impl_;
    Time            lastTime_;
    Step            stepCount_;
    LoggingPolicy   policy_;
};

} // namespace libecs

/** @} */ // logging module


#endif /* __LOGGER_HPP */

