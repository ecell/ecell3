//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-CELL is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-CELL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-CELL -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Eiichiro Adachi
//


#ifndef __DYNAMICPRIORITYQUEUE_HPP
#define __DYNAMICPRIORITYQUEUE_HPP
#include <vector>
#include <algorithm>


template < typename key_type >
class DynamicPriorityQueue
{
  
  template < class T >
  struct PtrGreater
  {
    bool operator()( T x, T y ) const { return *y < *x; }
  };



public:

  typedef std::vector< key_type >    KeyVector;
  typedef std::vector< key_type* >   KeyPtrVector;

  typedef typename KeyVector::size_type       size_type;
  typedef typename KeyVector::difference_type index_type;

  typedef std::vector< index_type >  IndexVector;


  DynamicPriorityQueue();
  
  void changeOneKey( index_type aPosition, key_type aKey );

  void changeTopKey( key_type aKey );

  index_type topIndex() const 
  {
    return( c.front() - theFirstKeyPtr );
  }

  const key_type& top() const
  {
    return *( c.front() );
  }

  void pop();
  void push( key_type aKey );

  bool empty() const
  {
    return ( size() == 0 );
  }

  size_type size() const
  {
    return theSize;
  }

private:

  void goUp( index_type );
  void goDown( index_type );


private:

  IndexVector  theIndices;
  KeyVector    v;
  KeyPtrVector c;

  key_type*  theFirstKeyPtr;
  index_type theSize;

  PtrGreater< key_type* > comp;

};



// begin implementation

template < typename key_type >
DynamicPriorityQueue< key_type >::DynamicPriorityQueue()
  :
  theSize( 0 )
{
  ; // do nothing
}


template < typename key_type >
void DynamicPriorityQueue< key_type >::
changeOneKey( index_type aPosition, key_type aNewKey )
{
  const index_type anIndex( theIndices[aPosition] );
  assert( anIndex < size() );

  if( *c[anIndex] != aNewKey )
    {
      if( comp( &aNewKey, c[anIndex] ) )
	{
	  *c[anIndex] = aNewKey;
	  goDown( anIndex );
	}
      else
	{
	  *c[anIndex] = aNewKey;
	  goUp( anIndex );
	}
    }
}


template < typename key_type >
void DynamicPriorityQueue<key_type>::goUp( index_type anIndex )
{
  index_type aPredecessor = ( anIndex - 1 ) / 2;
  key_type* anKeyPtr( c[anIndex] );

  while( aPredecessor != anIndex && comp( c[aPredecessor], anKeyPtr ) )
    {
      c[anIndex] = c[aPredecessor];
      theIndices[ c[anIndex] - theFirstKeyPtr ] = anIndex;
      anIndex = aPredecessor;
      aPredecessor = ( anIndex - 1 ) / 2;
    }

  c[anIndex] = anKeyPtr;
  theIndices[ c[anIndex] - theFirstKeyPtr ] = anIndex;
}


template < typename key_type >
void DynamicPriorityQueue< key_type >::goDown( index_type anIndex )
{
  index_type aSuccessor( ( anIndex + 1 ) * 2 - 1 );

  if( aSuccessor < size() - 1 && comp( c[aSuccessor], c[aSuccessor + 1] ) )
    {
      ++aSuccessor;
    }

  key_type* aKey( c[anIndex] );
  
  while( aSuccessor < size() && comp( aKey, c[aSuccessor] ) )
    {
      c[anIndex] = c[aSuccessor];
      theIndices[ c[anIndex] - theFirstKeyPtr ] = anIndex;
      anIndex = aSuccessor;
      aSuccessor = ( anIndex + 1 ) * 2 - 1;

      if( aSuccessor < size() - 1 && 
	  comp( c[aSuccessor], c[ aSuccessor + 1 ] ) )
	{
	  ++aSuccessor;
	}
    }

  c[anIndex] = aKey;
  theIndices[ c[anIndex] - theFirstKeyPtr ] = anIndex;
}


template < typename key_type >
void DynamicPriorityQueue< key_type >::pop()
{
  key_type* aKey( c[0] );
  --theSize;
  c[0] = c[size()];
  c[size()] = aKey;
  
  theIndices[ c[0] - theFirstKeyPtr ] = 0;
  
  goDown( 0 );
}


template < typename key_type >
void DynamicPriorityQueue< key_type >::push( key_type aKey )
{
  const index_type anOldSize( theSize );

  ++theSize;

  if( size() > c.size() )
    {
      c.resize( size() );
      v.resize( size() );
      theIndices.resize( size() );
      theFirstKeyPtr = v.begin().base();
    
      for( index_type i( 0 ); i < size(); ++i )
	{
	  c[i] = &v[i];
	}

      *c[ anOldSize ] = aKey;
 
      make_heap( c.begin(), c.end(), comp );

      for( index_type i( 0 ); i < size(); ++i )
	{
	  theIndices[ c[i] - theFirstKeyPtr ] = i;
	}
    }
  else
    {
      *c[ anOldSize ] = aKey;  
      if( comp( &aKey, c[ anOldSize ] ) )
	{
	  goDown( anOldSize );
	}
      else
	{
	  goUp( anOldSize ); 
	}
    }
}


template < typename key_type >
void DynamicPriorityQueue< key_type >::changeTopKey( key_type aNewKey )
{
  index_type anIndex( theIndices[ topIndex() ] );

  if( *c[anIndex] != aNewKey )
    {
      *c[anIndex] = aNewKey;
      goDown( anIndex );
    }
  //  changeOneKey(topIndex(),k);
}


#endif // __DYNAMICPRIORITYQUEUE_HPP



/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/





