//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2014 Keio University
//       Copyright (C) 2008-2014 RIKEN
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
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#ifndef __PROCESSEVENT_HPP
#define __PROCESSEVENT_HPP

#include "libecs/Defs.hpp"
#include "libecs/Process.hpp"

#include "libecs/EventScheduler.hpp"

namespace libecs
{

class LIBECS_API ProcessEvent: public EventBase
{
public:
    ProcessEvent( Time aTime = 0, Process* aProcess = 0)
        : EventBase( aTime ),
          theProcess( aProcess )
    {
        ; // do nothing
    }


    const Process* getProcess() const
    {
        return theProcess;
    }


    void fire()
    {
        //FIXME: should be theProcess->fire();
        theProcess->addValue( 1.0 );

        reschedule( getTime() );
    }

    void update( Time aTime )
    {
        reschedule( aTime );
    }


    void reschedule( Time aTime )
    {
        const Time aNewStepInterval( theProcess->getStepInterval() );
        setTime( aNewStepInterval + aTime );
    }


    const bool isDependentOn( ProcessEvent const& anEvent ) const
    {
        return theProcess->isDependentOn( anEvent.getProcess() );
    }


    const bool operator< ( ProcessEvent const& rhs ) const
    {
        if( getTime() > rhs.getTime() )
        {
            return false;
        }
        if( getTime() < rhs.getTime() )
        {
            return true;
        }
        if( theProcess->getPriority() < rhs.getProcess()->getPriority() )
        {
            return true;
        }
        return false;
    }


    const bool operator> ( ProcessEvent const& rhs ) const
    {
        if( getTime() < rhs.getTime() )
        {
            return false;
        }
        if( getTime() > rhs.getTime() )
        {
            return true;
        }
        if( theProcess->getPriority() > rhs.getProcess()->getPriority() )
        {
            return true;
        }
        return false;
    }


    const bool operator<= ( ProcessEvent const& rhs ) const
    {
        return !( *this > rhs );
    }


    const bool operator>= ( ProcessEvent const& rhs ) const
    {
        return !( *this < rhs );
    }


    const bool operator!= ( ProcessEvent const& rhs ) const
    {
        return getTime() != rhs.getTime() ||
                getProcess() != rhs.getProcess();
    }


private:

    Process* theProcess;

};

} // namespace libecs

#endif /* __PROCESSEVENT_HPP */
