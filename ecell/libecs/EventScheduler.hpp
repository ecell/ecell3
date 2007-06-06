//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2007 Keio University
//       Copyright (C) 2005-2007 The Molecular Sciences Institute
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

#ifndef __EVENTSCHEDULER_HPP
#define __EVENTSCHEDULER_HPP

#include "libecs.hpp"
#include "DynamicPriorityQueue.hpp"
#include "AssocVector.h"

namespace libecs
{

  /** @addtogroup model The Model.

      The model.

      @ingroup libecs
      @{ 
   */ 

  /** @file */

  DECLARE_CLASS( EventBase );


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

  class EventBase
  {

  public:

    EventBase( TimeParam aTime )
      :
      theTime( aTime )
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

   

    const bool operator< ( EventBaseCref rhs ) const
    {
      if( getTime() < rhs.getTime() )
	{
	  return true;
	}
      else
	{
	  return false;
	}
    }

    const bool operator> ( EventBaseCref rhs ) const
    {
      if( getTime() > rhs.getTime() )
	{
	  return true;
	}
      else
	{
	  return false;
	}
    }

    const bool operator>= ( EventBaseCref rhs ) const
    {
        return !(*this < rhs);
    }

    const bool operator<= ( EventBaseCref rhs ) const
    {
        return !(*this > rhs);
    }

    const bool operator!= ( EventBaseCref rhs ) const
    {
      if( getTime() == rhs.getTime() )
	{
	  return false;
	}
      else
	{
	  return true;
	}
    }


    // dummy, because DynamicPriorityQueue requires this. better without.
    EventBase()
    {
      ; // do nothing
    }


  private:

    Time             theTime;
  };


  /**
     Event scheduler.

     This class works as a sequential
     event scheduler with a heap-tree based priority queue.

  */

  template <class Event_>
  class EventScheduler
  {

  public:

    typedef Event_ Event;
    typedef DynamicPriorityQueue<Event, VolatileIDPolicy> EventPriorityQueue;

    typedef typename EventPriorityQueue::Index EventIndex;
    typedef typename EventPriorityQueue::ID EventID;

    typedef std::vector<EventID> EventIDVector;
    typedef Loki::AssocVector<EventID, EventIDVector> EventIDVectorMap;

    EventScheduler()
    {
      ; // do nothing
    }

    ~EventScheduler()
    {
      ; // do nothing
    }


    const EventIndex getSize() const
    {
      return theEventPriorityQueue.getSize();
    }

    const Event& getTopEvent() const
    {
      return theEventPriorityQueue.getTop();
    }

    Event& getTopEvent()
    {
      return theEventPriorityQueue.getTop();
    }

    EventID getTopID()
    {
      return theEventPriorityQueue.getTopID();
    }

    const Event& getEvent( const EventID anID ) const
    {
      return theEventPriorityQueue.get( anID );
    }

    Event& getEvent( const EventID anID )
    {
      return theEventPriorityQueue.get( anID );
    }

    void step()
    {
      Event& aTopEvent( theEventPriorityQueue.getTop() );
      const Time aCurrentTime( aTopEvent.getTime() );
      const EventID aTopEventID( getTopID() );

      // fire top
      aTopEvent.fire();
      theEventPriorityQueue.moveDown( aTopEventID );

      // update dependent events
      const EventIDVector&
	anEventIDVector( theEventDependencyMap[ aTopEventID ] );

      for( typename EventIDVector::const_iterator 
	     i( anEventIDVector.begin() );
	   i != anEventIDVector.end(); ++i )
	{
	  updateEvent( *i, aCurrentTime );
	}
    }

    void updateAllEvents( TimeParam aCurrentTime )
    {
      typedef typename EventPriorityQueue::IDIterator IDIterator;
      for( IDIterator i( theEventPriorityQueue.begin() );
	   i != theEventPriorityQueue.end(); ++i )
	{
	  updateEvent( *i, aCurrentTime );
	}
    }

    void updateEvent( const EventID anID, TimeParam aCurrentTime )
    {
      Event& anEvent( theEventPriorityQueue.get( anID ) );
      const Time anOldTime( anEvent.getTime() );
      anEvent.update( aCurrentTime );
      const Time aNewTime( anEvent.getTime() );

      // theEventPriorityQueue.move( anIndex );
      if( aNewTime >= anOldTime )
	{
	  theEventPriorityQueue.moveDown( anID );
	}
      else
	{
	  theEventPriorityQueue.moveUp( anID );
	}
    }


    void updateEventDependency();  // update all

    void updateEventDependency( const EventID anID );
    
    void clear()
    {
      theEventPriorityQueue.clear();
      theEventDependencyMap.clear();
    }

    const EventID addEvent( const Event& anEvent )
    {
      return theEventPriorityQueue.push( anEvent );
    }


    // this is here for DiscreteEventStepper::log().
    // should be removed in future. 
    const EventIDVector& getDependencyVector( const EventID& anID )
    {
      return theEventDependencyMap[ anID ] ;
    }

  private:

    EventPriorityQueue theEventPriorityQueue;
    EventIDVectorMap theEventDependencyMap;
  };

  


  template < class Event >
  void EventScheduler<Event>::updateEventDependency()
  {
    typedef typename EventPriorityQueue::IDIterator IDIterator;
    for( IDIterator i( theEventPriorityQueue.begin() );
	 i != theEventPriorityQueue.end(); ++i )
      {
	updateEventDependency( *i );
      }
  }

  template < class Event >
  void EventScheduler<Event>::
  updateEventDependency( const EventID i1 )
  {
    typedef typename EventPriorityQueue::IDIterator IDIterator;
    const Event& anEvent1( theEventPriorityQueue.get( i1 ) );

    EventIDVector& anEventIDVector( theEventDependencyMap[ i1 ] );
    anEventIDVector.clear();

    for( IDIterator i( theEventPriorityQueue.begin() );
         i != theEventPriorityQueue.end(); ++i )
      {
	if( i1 == *i )
	  {
	    // don't include itself
	    continue;
	  }
	
	const Event& anEvent2( theEventPriorityQueue.get( *i ) );
	
	if( anEvent2.isDependentOn( anEvent1 ) )
	  {
	    anEventIDVector.push_back( *i );
	  }
      }
    
    std::sort( anEventIDVector.begin(), anEventIDVector.end() );
  }





  /*@}*/

} // namespace libecs




#endif /* __EVENTSCHEDULER_HPP */




/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/

