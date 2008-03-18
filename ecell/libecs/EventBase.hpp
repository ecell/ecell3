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

#ifndef __EVENTBASE_HPP
#define __EVENTBASE_HPP

/**
   @addtogroup model The Model.

   The model.

   @ingroup libecs
 */
/** @{ */

namespace libecs {

/**
   EventBase

   A subclass must define three customization points;

   void fire()
   {
     (1) do what this event is supposed to do.
     (2) setTime( next scheduled time of this event );
   }

   void update( const Event& anEvent )
   {
     Given the last fired Event (anEvent) that this Event
     depends on,

     (1) recalculate scheduled time (if necessary).
     (2) setTime( new scheduled time ).
   }

   const bool isDependentOn( const Event& anEvent )
   {
     Return true if this Event must be updated when the
     given Event (anEvent) fired.  Otherwise return false;
   }
*/

class LIBECS_API EventBase
{
public:
    EventBase( TimeParam aTime )
        : theTime( aTime )
    {
        ; // do nothing
    }

    void setTime( TimeParam aTime )
    {
        theTime = aTime;
    }

    const Time getTime() const
    {
        return theTime;
    }

    bool operator<( const EventBase& rhs ) const
    {
        if ( getTime() < rhs.getTime() )
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    bool operator>( const EventBase& rhs ) const
    {
        if ( getTime() > rhs.getTime() )
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    bool operator>=( const EventBase& rhs ) const
    {
        return !operator<( rhs );
    }

    bool operator<=( const EventBase& rhs ) const
    {
        return !operator>( rhs );
    }

    bool operator==( const EventBase& rhs ) const
    {
        return getTime() == rhs.getTime();
    }

    bool operator!=( const EventBase& rhs ) const
    {
        return !operator==( rhs );
    }

private:
    Time theTime;
};

} // namespace libecs

/** @} */

#endif /* __EVENTBASE_HPP */
/*
  Do not modify
  $Author: moriyoshi $
  $Revision: 3085 $
  $Locker$
*/
