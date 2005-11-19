//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 1996-2005 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//

#ifndef __EVENTSCHEDULER_HPP
#define __EVENTSCHEDULER_HPP

#include "libecs.hpp"
#include "DynamicPriorityQueue.hpp"

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

     void update( EventCref anEvent )
     {
       Given the last fired Event (anEvent) that this Event
       depends on,

       (1) recalculate scheduled time (if necessary).
       (2) setTime( new scheduled time ).
     }

     const bool isDependentOn( EventCref anEvent )
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

    DECLARE_TYPE( Event_, Event );

    DECLARE_TYPE( DynamicPriorityQueue<Event>, EventPriorityQueue );

    typedef typename DynamicPriorityQueue<Event>::Index EventIndex_;
    DECLARE_TYPE( EventIndex_, EventIndex );

    typedef std::vector<EventIndex> EventIndexVector;
    typedef std::vector<EventIndexVector> EventIndexVectorVector;

    //DECLARE_VECTOR( EventIndex, EventIndexVector );
    //DECLARE_VECTOR( EventIndexVector, EventIndexVectorVector );


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

    EventCref getTopEvent() const
    {
      return theEventPriorityQueue.getTopItem();
    }

    EventRef getTopEvent()
    {
      return theEventPriorityQueue.getTopItem();
    }

    EventIndex getTopIndex()
    {
      return theEventPriorityQueue.getTopIndex();
    }

    EventCref getEvent( const EventIndex anIndex ) const
    {
      return theEventPriorityQueue.getItem( anIndex );
    }

    EventRef getEvent( const EventIndex anIndex )
    {
      return theEventPriorityQueue.getItem( anIndex );
    }

    void step()
    {
      EventRef aTopEvent( theEventPriorityQueue.getTopItem() );
      const Time aCurrentTime( aTopEvent.getTime() );
      const EventIndex aTopEventIndex( getTopIndex() );

      // fire top
      aTopEvent.fire();
      theEventPriorityQueue.moveDown( aTopEventIndex );

      // update dependent events
      const EventIndexVector&
	anEventIndexVector( theEventDependencyArray[ aTopEventIndex ] );

      for( typename EventIndexVector::const_iterator 
	     i( anEventIndexVector.begin() );
	   i != anEventIndexVector.end(); ++i )
	{
	  const EventIndex anIndex( *i );

	  updateEvent( anIndex, aCurrentTime );
	}
    }

    void updateAllEvents( TimeParam aCurrentTime )
    {
      const EventIndex aSize( getSize() );
      for( EventIndex anIndex( 0 ); anIndex != aSize; ++anIndex )
	{
	  updateEvent( anIndex, aCurrentTime );
	}
    }

    void updateEvent( const EventIndex anIndex, TimeParam aCurrentTime )
    {
      EventRef anEvent( theEventPriorityQueue.getItem( anIndex ) );
      const Time anOldTime( anEvent.getTime() );
      anEvent.update( aCurrentTime );
      const Time aNewTime( anEvent.getTime() );

      // theEventPriorityQueue.move( anIndex );
      if( aNewTime >= anOldTime )
	{
	  theEventPriorityQueue.moveDown( anIndex );
	}
      else
	{
	  theEventPriorityQueue.moveUp( anIndex );
	}
    }


    void updateEventDependency();  // update all

    void updateEventDependency( const EventIndex anIndex );
    
    void clear()
    {
      theEventPriorityQueue.clear();
      theEventDependencyArray.clear();
    }

    const EventIndex addEvent( EventCref anEvent )
    {
      return theEventPriorityQueue.pushItem( anEvent );
    }


    // this is here for DiscreteEventStepper::log().
    // should be removed in future. 
    const EventIndexVector& getDependencyVector( const EventIndex anIndex )
    {
      return theEventDependencyArray[ anIndex ] ;
    }

  private:

    EventPriorityQueue       theEventPriorityQueue;

    EventIndexVectorVector   theEventDependencyArray;

  };

  


  template < class Event >
  void EventScheduler<Event>::updateEventDependency()
  {
    theEventDependencyArray.resize( theEventPriorityQueue.getSize() );
    
    for( EventIndex i1( 0 ); i1 != theEventPriorityQueue.getSize(); ++i1 )
      {
	updateEventDependency( i1 );
      }
  }

  template < class Event >
  void EventScheduler<Event>::
  updateEventDependency( const EventIndex i1 )
  {
    EventCref anEvent1( theEventPriorityQueue.getItem( i1 ) );

    EventIndexVector& anEventIndexVector( theEventDependencyArray[ i1 ] );
    anEventIndexVector.clear();

    for( EventIndex i2( 0 ); i2 < theEventPriorityQueue.getSize(); ++i2 )
      {
	if( i1 == i2 )
	  {
	    // don't include itself
	    continue;
	  }
	
	EventCref anEvent2( theEventPriorityQueue.getItem( i2 ) );
	
	if( anEvent2.isDependentOn( anEvent1 ) )
	  {
	    anEventIndexVector.push_back( i2 );
	  }
      }
    
    std::sort( anEventIndexVector.begin(), anEventIndexVector.end() );
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

