//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
//                Copyright (C) 2005 The Molecular Sciences Institute
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
// written by Eiichiro Adachi
// modified by Koichi Takahashi
//


#ifndef __DYNAMICPRIORITYQUEUE_HPP
#define __DYNAMICPRIORITYQUEUE_HPP
#include <vector>
#include <algorithm>

//#include "Util.hpp"

namespace libecs
{

template < typename Item >
class DynamicPriorityQueue
{
  

public:

  typedef std::vector< Item >    ItemVector;
  typedef std::vector< Item* >   ItemPtrVector;

  typedef typename ItemVector::size_type       size_type;
  typedef typename ItemVector::difference_type difference_type;
  typedef size_type                            Index;

  typedef std::vector< Index >  IndexVector;


  DynamicPriorityQueue();
  
  inline void move( const Index anIndex );

  inline void moveTop();

  const Index getTopIndex() const 
  {
    return( getItemIndex( theItemPtrVector.front() ) );
  }

  const Item& getTopItem() const
  {
    return *( theItemPtrVector.front() );
  }

  Item& getTopItem()
  {
    return *( theItemPtrVector.front() );
  }

  const Item& getItem( const Index anIndex ) const
  {
    return theItemVector[ anIndex ];
  }

  Item& getItem( const Index anIndex )
  {
    return theItemVector[ anIndex ];
  }

  void popItem();
  const Index pushItem( const Item& anItem )
  {
    const Index anOldSize( theSize );
    
    ++theSize;
    
    if( getSize() > theItemPtrVector.size() )
      {
	theItemVector.resize( getSize() );
	theItemPtrVector.resize( getSize() );
	theIndexVector.resize( getSize() );
	
	theItemVector.push_back( anItem );

	for( Index i( 0 ); i < getSize(); ++i )
	  {
	    theItemPtrVector[i] = &theItemVector[i];
	  }

	*theItemPtrVector[ anOldSize ] = anItem;
 
	make_heap( theItemPtrVector.begin(), theItemPtrVector.end(), comp );

	for( Index i( 0 ); i < getSize(); ++i )
	  {
	    theIndexVector[ getItemIndex( theItemPtrVector[i] ) ] = i;
	  }
      }
    else
      {
	*theItemPtrVector[ anOldSize ] = anItem;  
	 moveUpPos( anOldSize ); 
      }

    return anOldSize;
  }


  bool isEmpty() const
  {
    return ( getSize() == 0 );
  }

  size_type getSize() const
  {
    return theSize;
  }


  void clear();

  void moveUp( const Index anIndex )
  {
    const Index aPosition( theIndexVector[anIndex] );
    moveUpPos( aPosition );
  }


  void moveDown( const Index anIndex )
  {
    const Index aPosition( theIndexVector[anIndex] );
    moveDownPos( aPosition );
  }

private:

  inline void moveUpPos( const Index aPosition );
  inline void moveDownPos( const Index aPosition );

  /*
    This method returns the index of the given pointer to Item.

    The pointer must point to a valid item on theItemVector.
    Returned index is that of theItemVector.
  */
  const Index getItemIndex( const Item * const ItemPtr ) const
  {
    // this cast is safe.
    return static_cast< Index >( ItemPtr - &theItemVector.front() );
  }

private:

  ItemVector    theItemVector;
  ItemPtrVector theItemPtrVector;
  IndexVector   theIndexVector;

  Index    theSize;

  struct PtrGreater
  {
    bool operator()( Item* const x, Item* const y ) const { return *y < *x; }
  };

  PtrGreater comp;

};



// begin implementation

template < typename Item >
DynamicPriorityQueue< Item >::DynamicPriorityQueue()
  :
  theSize( 0 )
{
  ; // do nothing
}


template < typename Item >
void DynamicPriorityQueue< Item >::clear()
{
  theItemVector.clear();
  theItemPtrVector.clear();
  theIndexVector.clear();
  
  theSize = 0;
  
}


template < typename Item >
inline void DynamicPriorityQueue< Item >::
move( Index anIndex )
{
  //  assert( aPosition < getSize() );
  const Index aPosition( theIndexVector[anIndex] );

  moveDownPos( aPosition );

  // If above moveDown() didn't move this item,
  // then we need to try moveUp() too.  If moveDown()
  // did work, nothing should be done.
  if( theIndexVector[anIndex] == aPosition )
    {
      moveUpPos( aPosition );
    }
}


template < typename Item >
void DynamicPriorityQueue<Item>::moveUpPos( Index aPosition )
{
  Item* const anItem( theItemPtrVector[ aPosition ] );

  // main loop
  while( aPosition > 0 )
    {
      Index aPredecessor( ( aPosition - 1 ) / 2 );
      Item* aPredItem( theItemPtrVector[ aPredecessor ] );

      if( comp( anItem, aPredItem ) )
        {
          break;
        }

      theItemPtrVector[aPosition] = aPredItem;
      theIndexVector[ getItemIndex( aPredItem ) ] = aPosition;
      aPosition = aPredecessor;
    }

  theItemPtrVector[aPosition] = anItem;
  theIndexVector[ getItemIndex( anItem ) ] = aPosition;
}

// this is an optimized version.
template < typename Item >
void DynamicPriorityQueue< Item >::moveDownPos( Index aPosition )
{
  Item* const anItem( theItemPtrVector[aPosition] );

  // main loop
  while( 1 )
    {
      Index aSuccessor( aPosition * 2 + 1 );

      if( aSuccessor >= getSize() )
        {
          break;
        }

      if( aSuccessor + 1 < getSize() &&
          comp( theItemPtrVector[ aSuccessor ],
                theItemPtrVector[ aSuccessor + 1 ] ) )
        {
          ++aSuccessor;
        }

      Item* const aSuccItem( theItemPtrVector[ aSuccessor ] );

      // if the going down is finished, break.
      if( !comp( anItem, aSuccItem ) )
	{
	  break;
	}
      // bring up the successor
      theItemPtrVector[aPosition] = aSuccItem;
      theIndexVector[ getItemIndex( aSuccItem ) ] = aPosition;
      aPosition = aSuccessor;
    }

  theItemPtrVector[aPosition] = anItem;
  theIndexVector[ getItemIndex( anItem ) ] = aPosition;
}

template < typename Item >
void DynamicPriorityQueue< Item >::popItem()
{
  if( this->theSize == 0 )
    {
      return;
    }
  --theSize;

  Item* anItem( theItemPtrVector[0] );
  theItemPtrVector[0] = theItemPtrVector[getSize()];
  theItemPtrVector[getSize()] = anItem;
  
  theIndexVector[ getItemIndex( theItemPtrVector[0] ) ] = 0;
  
  moveDownPos( 0 );
}

template < typename Item >
void DynamicPriorityQueue< Item >::moveTop()
{
  moveDownPos( 0 );
}

} // namespace libecs

#endif // __DYNAMICPRIORITYQUEUE_HPP



/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/





