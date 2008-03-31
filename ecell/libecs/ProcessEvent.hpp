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
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#ifndef __PROCESSEVENT_HPP
#define __PROCESSEVENT_HPP

#include "libecs.hpp"
#include "Process.hpp"

#include "EventBase.hpp"

/** @file */

namespace libecs
{

DECLARE_CLASS( ProcessEvent );

class LIBECS_API ProcessEvent: public EventBase
{
public:
    ProcessEvent( TimeParam aTime, Process* process )
        : EventBase( aTime ), process_( process )
    {
        ; // do nothing
    }

    const Process* getProcess() const
    {
        return process_;
    }


    void fire()
    {
        //FIXME: should be process_->fire();
        process_->addValue( 1.0 );

        reschedule( getTime() );
    }

    void update( TimeParam aTime )
    {
        reschedule( aTime );
    }

    void reschedule( TimeParam aTime )
    {
        const Time aNewStepInterval( process_->getStepInterval() );
        setTime( aNewStepInterval + aTime );
    }

    const bool isDependentOn( const ProcessEvent& anEvent ) const
    {
        return process_->isDependentOn( anEvent.getProcess() );
    }


    const bool operator< ( const ProcessEvent& rhs ) const
    {
        // return theTime < rhs.theTime;

        if ( getTime() > rhs.getTime() )
        {
            return false;
        }
        if ( getTime() < rhs.getTime() )
        {
            return true;
        }
        if ( process_->getPriority() < rhs.getProcess()->getPriority() )
        {
            return true;
        }
        return false;
    }


    const bool operator> ( const ProcessEvent& rhs ) const
    {
        // return theTime < rhs.theTime;

        if ( getTime() < rhs.getTime() )
        {
            return false;
        }
        if ( getTime() > rhs.getTime() )
        {
            return true;
        }
        if ( process_->getPriority() > rhs.getProcess()->getPriority() )
        {
            return true;
        }
        return false;
    }


    const bool operator<= ( const ProcessEvent& rhs ) const
    {
        return !( *this > rhs );
    }


    const bool operator>= ( const ProcessEvent& rhs ) const
    {
        return !( *this < rhs );
    }


    const bool operator!= ( const ProcessEvent& rhs ) const
    {
        if ( getTime() == rhs.getTime() &&
                getProcess() == rhs.getProcess() )
        {
            return false;
        }
        else
        {
            return true;
        }
    }

private:
    Process* process_;
};

} // namespace libecs

/** @} */


#endif /* __PROCESSEVENT_HPP */
/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
