//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 1996-2007 Keio University
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

#include "Exceptions.hpp"

namespace libecs
{

template < typename Item >
class DynamicPriorityQueueTest;

template < typename Item >
class DynamicPriorityQueue
{
  friend class DynamicPriorityQueueTest< Item >;

public:

  typedef std::vector< Item >    ItemVector;
  typedef std::vector< Item* >   ItemPtrVector;

  typedef typename ItemVector::size_type       size_type;
  typedef typename ItemVector::difference_type difference_type;
  typedef size_type                            Index;

  typedef std::vector< Index >  IndexVector;


  DynamicPriorityQueue();

  void move( const Index anIndex )
  {
    const Index aPosition( theIndexVector[anIndex] );

    if ( aPosition == 0 )
      {
        moveDownPos( aPosition );
        return;
      }

    const Index aPredecessor( ( aPosition - 1 ) / 2 );

    if ( comp( theItemPtrVector[ aPredecessor ],
               theItemPtrVector[ aPosition ] ) )
      {
        moveUpPos( aPosition );
      }
    else 
      {
        moveUpPos( aPosition );
      }
  }

  const Index getTopIndex() const 
  {
    return( getIndex( theItemPtrVector.front() ) );
  }

  const Item& getTop() const
  {
    return *( theItemPtrVector.front() );
  }

  Item& getTop()
  {
    return *( theItemPtrVector.front() );
  }

  const Item& get( const Index anIndex ) const
  {
    return theItemVector[ anIndex ];
  }

  Item& get( const Index anIndex )
  {
    return theItemVector[ anIndex ];
  }

  Item popByPosition( const Index aPosition );

  Item pop( const Index anIndex )
  {
    return popByPosition( theIndexVector[ anIndex ] );
  }

  Item popTop()
  {
    return popByPosition( 0 );
  }

  const Index push( const Item& anItem )
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
 
	std::make_heap( theItemPtrVector.begin(), theItemPtrVector.end(), comp );

	for( Index i( 0 ); i < getSize(); ++i )
	  {
	    theIndexVector[ getIndex( theItemPtrVector[i] ) ] = i;
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

  void replace( const Index anIndex, const Item& anItem )
  {
    pop( anIndex );
    push( anItem );
  }

  void replaceTop( const Item& anItem )
  {
    popTop();
    push( anItem );
  }

private:

  void moveUpPos( const Index aPosition );
  void moveDownPos( const Index aPosition );

  /*
    This method returns the index of the given pointer to Item.

    The pointer must point to a valid item on theItemVector.
    Returned index is that of theItemVector.
  */
  const Index getIndex( const Item * const ItemPtr ) const
  {
    // this cast is safe.
    return static_cast< Index >( ItemPtr - &theItemVector.front() );
  }

  template< typename RandomAccessIterator, typename PosteriorityPredicate >
  static bool is_heap( RandomAccessIterator first,
                       RandomAccessIterator last,
                       PosteriorityPredicate former_is_less )
  {
    return is_heap( first, last, 0, former_is_less );
  }

  template< typename RandomAccessIterator, typename PosteriorityPredicate >
  static bool is_heap( RandomAccessIterator first,
                       RandomAccessIterator last,
                       typename RandomAccessIterator::difference_type pos,
                       PosteriorityPredicate former_is_less )
  {
    typename RandomAccessIterator::difference_type
      left_node_pos( pos * 2 + 1 ),
      right_node_pos( pos * 2 + 2 );

    if ( first >= last )
      {
        return true;
      }

    typename RandomAccessIterator::value_type node_value( *( first + pos ) );

    if ( first + left_node_pos >= last )
      {
        return true;
      }

    if ( former_is_less( node_value, *( first + left_node_pos ) ) )
      {
        fprintf( stderr, "%lf %lf\n",
                 (double) *node_value,
                 (double) **( first + left_node_pos ) );
        return false;
      }

    if ( first + right_node_pos >= last )
      {
        return true;
      }

    if ( former_is_less( node_value, *( first + right_node_pos ) ) )
      {
        fprintf( stderr, "%lf %lf\n",
                 (double) *node_value,
                 (double) **( first + right_node_pos ) );
        return false;
      }

    return is_heap< RandomAccessIterator, PosteriorityPredicate >(
                first, last, left_node_pos, former_is_less ) &&
           is_heap< RandomAccessIterator, PosteriorityPredicate >(
                first, last, right_node_pos, former_is_less );
  }

  const bool checkConsistency() const
  {
    return is_heap( theItemPtrVector.begin(),
                    theItemPtrVector.begin() + getSize(),
                    comp );
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
inline void DynamicPriorityQueue<Item>::moveUpPos( Index aPosition )
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
      theIndexVector[ getIndex( aPredItem ) ] = aPosition;
      aPosition = aPredecessor;
    }

  theItemPtrVector[aPosition] = anItem;
  theIndexVector[ getIndex( anItem ) ] = aPosition;
}

// this is an optimized version.
template < typename Item >
inline void DynamicPriorityQueue< Item >::moveDownPos( Index aPosition )
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
      theIndexVector[ getIndex( aSuccItem ) ] = aPosition;
      aPosition = aSuccessor;
    }

  theItemPtrVector[aPosition] = anItem;
  theIndexVector[ getIndex( anItem ) ] = aPosition;
}

template < typename Item >
inline Item DynamicPriorityQueue< Item >::popByPosition( const Index aPosition )
{
  if( this->theSize == 0 )
    {
      throw IllegalOperation( "DynamicPriorityQueue<>::popByPosition()",
                              "Queue is empty" );
    }
  --theSize;

  Item* anItem( theItemPtrVector[ aPosition ] );
  theItemPtrVector[ aPosition ] = theItemPtrVector[ theSize ];
  theItemPtrVector[ theSize ] = anItem;
  
  theIndexVector[ getIndex( anItem ) ] = 0;
  
  moveDownPos( aPosition );

  return *anItem;
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





