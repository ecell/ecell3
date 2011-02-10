//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2010 Keio University
//       Copyright (C) 2005-2009 The Molecular Sciences Institute
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

#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>

#include "libecs/Defs.hpp"
#include "libecs/LoggerAdapter.hpp"
#include "libecs/PhysicalLogger.hpp"
#include "libecs/DataPointVector.hpp"
#include "Exceptions.hpp"

namespace libecs
{

/**
    Logger module for logging and retrieving data runtime.
*/
class LIBECS_API Logger
{
public:
    typedef PhysicalLogger::size_type size_type;
    
    class Policy
    {
    public:
        Policy( Integer aMinimumStep = 1,
                Real    aMinimumTimeInterval = 0.0,
                bool    _continueOnError = false,
                Integer aMaxSpace = 0 )
            : theMinimumStep( aMinimumStep ),
              theMinimumTimeInterval( aMinimumTimeInterval ),
              continueOnError( _continueOnError ),
              theMaxSpace( aMaxSpace )
        {
            if( aMinimumTimeInterval < 0 )
            {
                THROW_EXCEPTION( ValueError,
                                 "negative value not allowed for minimum time "
                                 "interval");
            }
            if ( aMinimumStep < 0 )
            {
                THROW_EXCEPTION( ValueError,
                                 "negative value not allowed for minimum step");
            }
            if ( aMaxSpace < 0 )
            {
                THROW_EXCEPTION( ValueError,
                                 "invalid value for max space" );
            }
        }

        Integer getMinimumStep() const
        {
            return theMinimumStep;
        }

        void setMinimumStep( Integer aMinimumStep )
        {
            if ( aMinimumStep < 0 )
            {
                THROW_EXCEPTION( ValueError,
                                 "negative value not allowed for minimum step");
            }
            theMinimumStep = aMinimumStep;
        }

        Real getMinimumTimeInterval() const
        {
            return theMinimumTimeInterval;
        }

        void setMinimumTimeInterval( Real aMinimumTimeInterval )
        {
            if( aMinimumTimeInterval < 0 )
            {
                THROW_EXCEPTION( ValueError,
                                 "negative value not allowed for minimum time "
                                 "interval");
            }
            theMinimumTimeInterval = aMinimumTimeInterval;
        }

        bool doesContinueOnError() const
        {
            return continueOnError;
        }

        void setContinueOnError( bool _continueOnError )
        {
            continueOnError = _continueOnError;
        }

        Integer getMaxSpace() const
        {
            return theMaxSpace;
        }

        void setMaxSpace( Integer aMaxSpace )
        {
            if ( aMaxSpace < 0 )
            {
                THROW_EXCEPTION( ValueError,
                                 "Invalid value for max space" );
            }
            theMaxSpace = aMaxSpace;
        }

        Policy const& operator=( Policy const& rhs )
        {
            theMinimumStep         = rhs.theMinimumStep;
            theMinimumTimeInterval = rhs.theMinimumTimeInterval;
            continueOnError        = rhs.continueOnError;
            theMaxSpace            = rhs.theMaxSpace;
            return *this;
        }

    private:
        Integer theMinimumStep;
        Real theMinimumTimeInterval;
        bool continueOnError;
        Integer theMaxSpace;
    };

public:

    /**
         Constructor.

         Takes up the ownership of the given LoggerAdapter.

    */

    Logger( LoggerAdapter* aLoggerAdapter, Policy const& aPolicy = Policy() );

    /// Destructor

    ~Logger( void );


    /**
    
    Sets logging policy that is a vector of 4 numerical values. 
    0 (int)    - minimum step size between logs
    1 (real) - minimum time interval between logs
    2 (int) - action to be taken when disk space runs out
    3 (int) - user set max disk space, if 0 nothing 
    
    */
    void setLoggerPolicy( Policy const& pol );

    /**

    Returns logging policy vector.

    */

    const Policy& getLoggerPolicy( void );

    /**

        Log current value that theLoggerAdapter gives with aTime.

    */

    void log( Real aTime );


    /**
         Returns contents of the whole logger.

    */

    boost::shared_ptr< DataPointVector > getData( void ) const;

    /**
         Returns a slice of the data from aStartTime to anEndTime.

    */

    boost::shared_ptr< DataPointVector > getData( Real aStartTime,
                                                  Real anEndTime ) const;

    /**
         Returns a summary of the data from aStartTime to anEndTime with
         intervals anInterval between data elements.
    */

    boost::shared_ptr< DataPointVector > getData( Real aStartTime,
                                                  Real anEndTime, 
                                                  Real anInterval ) const;
    


    /**
         Returns time of the first element    in Logger.
    */

    Real getStartTime( void ) const;

    /**
         Returns time of the last element in Logger
    */

    Real getEndTime( void ) const;

    /**
        Returns size of logger
    */

    size_type getSize() const
    {
        return thePhysicalLogger.size();
    }


    /**
         This method does nothing as of version 3.1.103.
    */

    void flush()
    {
        ; // do nothing
    }


protected:

    /**
         Writes data (aTime, aValue ) onto the logger
    */

    void pushData( Real aTime, Real aValue )
    {
        thePhysicalLogger.push( DataPoint( aTime, aValue ) );
    }

    static boost::shared_ptr< DataPointVector > createEmptyVector();

private:

    /// no default constructor
    Logger( void );

    /// noncopyable
    Logger( Logger const& );


private:

    /// Data members

    PhysicalLogger                thePhysicalLogger;

    LoggerAdapter*                theLoggerAdapter;

    Integer                       theStepCounter;
    Real                          theLastTime;

    Policy                        thePolicy;
};

} // namespace libecs

#endif /* __LOGGER_HPP */

