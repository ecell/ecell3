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
// written by Koichi Takahashi based on the initial version by Eiichiro Adachi.
// modified by Moriyoshi Koizumi
//

#ifndef __DYNAMICPRIORITYQUEUE_HPP
#define __DYNAMICPRIORITYQUEUE_HPP

#include "Defs.hpp"

#include <functional>
#include <vector>
#include <algorithm>
#include <stdexcept>

#if defined( HAVE_UNORDERED_MAP )
#include <unordered_map>
#elif defined( HAVE_TR1_UNORDERED_MAP )
#include <tr1/unordered_map>
#else
#include <map>
#endif /* HAVE_UNORDERED_MAP */


namespace libecs
{


class PersistentIDPolicy
{

public:

    typedef uint64_t               ID;
    typedef std::vector< ID >      IDVector;
    typedef IDVector::size_type    Index;
    typedef IDVector::const_iterator IDIterator;

#if defined( HAVE_UNORDERED_MAP ) || defined( HAVE_TR1_UNORDERED_MAP )

    class IDHasher
        : 
        public std::unary_function<ID, std::size_t>
    {

    public:

        std::size_t operator()( ID value ) const
        {
            return static_cast<std::size_t>( value ) ^
                static_cast<std::size_t>( value >> ( sizeof( ID ) * 8 / 2 ) );
        }

    };

#endif // HAVE_UNORDERED_MAP || HAVE_TR1_UNORDERED_MAP

#if defined( HAVE_UNORDERED_MAP )
    typedef std::unordered_map<const ID, Index, IDHasher> IndexMap;
#elif defined( HAVE_TR1_UNORDERED_MAP )
    typedef std::tr1::unordered_map<const ID, Index, IDHasher> IndexMap;
#else 
    typedef std::map<const ID, Index> IndexMap;
#endif


    PersistentIDPolicy()
        :
        idCounter( 0 )
    {
        ; // do nothing
    }
    
    void reset()
    {
        idCounter = 0;
    }

    void clear()
    {
        this->idVector.clear();
        this->indexMap.clear();
    }

    const Index getIndex( const ID id ) const
    {
        IndexMap::const_iterator i = this->indexMap.find( id );

        if( i == this->indexMap.end() )
        {
            throw std::out_of_range( "PersistentIDPolicy::getIndex()" );
        }

        return (*i).second;
    }

    const ID getIDByIndex( const Index index ) const
    {
        return this->idVector[ index ];
    }

    const ID push( const Index index )
    {
        const ID id( this->idCounter );
        ++this->idCounter;

        this->indexMap.insert( IndexMap::value_type( id, index ) );
        this->idVector.push_back( id );

        return id;
    }

    void pop( const Index index )
    {
        // update the idVector and the indexMap.
        const ID removedID( this->idVector[ index ] );
        const ID movedID( this->idVector.back() );
        this->idVector[ index ] = movedID;
        this->idVector.pop_back();
        
        this->indexMap[ movedID ] = index;
        this->indexMap.erase( removedID );
    }

    IDIterator begin() const
    {
        return this->idVector.begin();
    }

    IDIterator end() const
    {
        return this->idVector.end();
    }

    const bool checkConsistency( const Index size ) const
    {
        if( this->idVector.size() != size )
        {
            return false;
        }

        if( this->indexMap.size() != size )
        {
            return false;
        }

        // assert correct mapping between the indexMap and the idVector.
        for( Index i( 0 ); i < size; ++i )
        {
            const ID id( this->idVector[i] );

            if (id >= this->idCounter)
            {
                return false;
            }

            IndexMap::const_iterator iter = this->indexMap.find( id );
            if (iter == this->indexMap.end())
            {
                return false;
            }

            if ((*iter).second != i)
            {
                return false;
            }
        }

        return true;
    }

private:

    // map itemVector index to id.
    IDVector      idVector;
    // map id to itemVector index.
    IndexMap      indexMap;

    ID   idCounter;

};

class VolatileIDPolicy
{
public:


    typedef size_t    Index;
    typedef Index     ID;

    class IDIterator
    {
    public:
        typedef ptrdiff_t difference_type;
        typedef VolatileIDPolicy::Index size_type;
        typedef VolatileIDPolicy::ID value_type;

    public:
        IDIterator( VolatileIDPolicy::Index _idx )
            : idx( _idx ) {}

        IDIterator operator++(int)
        {
            IDIterator retval = *this;
            ++this->idx;
            return retval;
        }

        IDIterator& operator++()
        {
            ++this->idx;
            return *this;
        }

        IDIterator operator--(int)
        {
            IDIterator retval = *this;
            --this->idx;
            return retval;
        }

        IDIterator& operator--()
        {
            --this->idx;
            return *this;
        }

        value_type operator*()
        {
            return this->idx;
        }

        difference_type operator-( const IDIterator& that )
        {
            return this->idx - that.idx;
        }

        bool operator==( const IDIterator& that )
        {
            return this->idx == that.idx;
        }

        bool operator!=( const IDIterator& that )
        {
            return this->idx != that.idx;
        }

        bool operator <( const IDIterator& that )
        {
            return this->idx < that.idx;
        }

        bool operator >( const IDIterator& that )
        {
            return this->idx > that.idx;
        }

    private:
        size_type idx;
    };

public:
    void reset()
    {
        ; // do nothing
    }

    void clear()
    {
        ; // do nothing
    }

    const Index getIndex( const ID id ) const
    {
        return id;
    }

    const ID getIDByIndex( const Index index ) const
    {
        return index;
    }

    const ID push( const Index index )
    {
        maxIndex = index + 1;
        return index;
    }

    void pop( const Index index )
    {
        BOOST_ASSERT( maxIndex == index + 1 );
        maxIndex = index;
    }

    const bool checkConsistency( const Index size ) const
    {
        return true;
    }

    IDIterator begin() const
    {
        return IDIterator( 0 );
    }

    IDIterator end() const
    {
        return IDIterator( maxIndex );
    }

private:
    Index maxIndex;
};

/**
   Dynamic priority queue for items of type Item.

   When IDPolicy is PersistentIDPolicy, IDs given to
   pushed items are persistent for the life time of this
   priority queue.

   When VolatileIDPolicy is used as the IDPolicy, IDs
   are valid only until the next call or pop or push methods.
   However, VolatileIDPolicy saves some memory and eliminates
   the overhead incurred in pop/push methods.

*/

template<typename T> class DynamicPriorityQueueTest;

template < typename Item, class IDPolicy = PersistentIDPolicy >
class DynamicPriorityQueue
{
    friend class DynamicPriorityQueueTest<Item>;

public:
    typedef std::vector< Item >    ItemVector;

    typedef typename IDPolicy::ID ID;
    typedef typename IDPolicy::Index Index;

    typedef std::vector< Index >   IndexVector;

    typedef typename IDPolicy::IDIterator IDIterator;

public:
    DynamicPriorityQueue();
  
    const bool isEmpty() const
    {
        return this->itemVector.empty();
    }

    const Index getSize() const
    {
        return this->itemVector.size();
    }

    void clear();

    Item& getTop()
    {
        return this->itemVector[ getTopIndex() ];
    }

    Item const& getTop() const
    {
        return this->itemVector[ getTopIndex() ];
    }

    Item& get( const ID id )
    {
        return this->itemVector[ this->pol.getIndex( id ) ];
    }

    Item const& get( const ID id ) const
    {
        return this->itemVector[ this->pol.getIndex( id ) ];
    }

    ID getTopID() const
    {
        return this->pol.getIDByIndex( getTopIndex() );
    }

    void popTop()
    {
        popByIndex( getTopIndex() );
    }

    void pop( const ID id )
    {
        popByIndex( this->pol.getIndex( id ) );
    }

    void replaceTop( const Item& item );

    void replace( const ID id, const Item& item );

    inline const ID push( const Item& item );

    void dump() const;

    Item& operator[]( const ID id )
    {
        return get( id );
    }

    Item const& operator[]( const ID id ) const
    {
        return get( id );
    }


    inline void popByIndex( const Index index );

    Item& getByIndex( const Index index )
    {
        return this->itemVector[ index ];
    }

    const Index getTopIndex() const 
    {
        return this->heap[0];
    }

    void move( const Index index )
    {
        const Index pos( this->positionVector[ index ] );
        movePos( pos );
    }

    inline void movePos( const Index pos );

    void moveTop()
    {
        moveDownPos( 0 );
    }

    void moveUpByIndex( const Index index )
    {
        const Index position( this->positionVector[ index ] );
        moveUpPos( position );
    }

    void moveUp( const ID id )
    {
        moveUpByIndex( pol.getIndex( id ) );
    }

    void moveDownByIndex( const Index index )
    {
        const Index position( this->positionVector[ index ] );
        moveDownPos( position );
    }

    void moveDown( const ID id )
    {
        moveDownByIndex( pol.getIndex( id ) );
    }

    IDIterator begin() const
    {
        return pol.begin();
    }

    IDIterator end() const
    {
        return pol.end();
    }

protected:
    // self-diagnostic method
    const bool checkConsistency() const;


private:

    inline void moveUpPos( const Index position, const Index start = 0 );
    inline void moveDownPos( const Index position );

private:

    ItemVector    itemVector;
    IndexVector   heap;

    // maps itemVector index to heap position.
    IndexVector   positionVector;

    std::less_equal< const Item > comp;

    IDPolicy pol;
};




template < typename Item, class IDPolicy >
DynamicPriorityQueue< Item, IDPolicy >::DynamicPriorityQueue()
{
    ; // do nothing
}


template < typename Item, class IDPolicy >
void DynamicPriorityQueue< Item, IDPolicy >::clear()
{
    this->itemVector.clear();
    this->heap.clear();
    this->positionVector.clear();
    this->pol.clear();
}


template < typename Item, class IDPolicy >
void DynamicPriorityQueue< Item, IDPolicy >::
movePos( const Index pos )
{
    const Index index( this->heap[ pos ] );
    const Item& item( this->itemVector[ index ] );

    const Index size( getSize() );

    if( pos < size / 2 )
    {
        const Index succ( 2 * pos + 1 );
        if( this->comp( this->itemVector[ this->heap[ succ ] ], item ) ||
            ( succ + 1 < size && 
              this->comp( this->itemVector[ this->heap[ succ + 1 ] ], 
                          item ) ) )
        {
            moveDownPos( pos );
            return;
        }
    }

    if( pos > 0 )
    {
        const Index pred( ( pos - 1 ) / 2 );
        if( this->comp( item, this->itemVector[ this->heap[ pred ] ] ) )
        {
            moveUpPos( pos );
            return;
        }
    }
}

template < typename Item, class IDPolicy >
void DynamicPriorityQueue< Item, IDPolicy >::moveUpPos( const Index position, 
                                                        const Index start )
{
    if ( position == 0 )
    {
        return;
    }

    const Index index( this->heap[ position ] );
    const Item& item( this->itemVector[ index ] );

    Index pos( position );
    while( pos > start )
    {
        const Index pred( ( pos - 1 ) / 2 );
        const Index predIndex( this->heap[ pred ] );
        if( this->comp( this->itemVector[ predIndex ], item ) )
        {
            break;
        }

        this->heap[ pos ] = predIndex;
        this->positionVector[ predIndex ] = pos;
        pos = pred;
    }

    this->heap[ pos ] = index;
    this->positionVector[ index ] = pos;
}


template < typename Item, class IDPolicy >
void 
DynamicPriorityQueue< Item, IDPolicy >::moveDownPos( const Index position )
{
    const Index index( this->heap[ position ] );
    const Item& item( this->itemVector[ index ] );

    const Index size( getSize() );
    
    Index succ( 2 * position + 1 );
    Index pos( position );
    while( succ < size )
    {
        const Index rightPos( succ + 1 );
        if( rightPos < size && 
            this->comp( this->itemVector[ this->heap[ rightPos ] ],
                        this->itemVector[ this->heap[ succ ] ] ) )
        {
            succ = rightPos;
        }

        this->heap[ pos ] = this->heap[ succ ];
        this->positionVector[ this->heap[ pos ] ] = pos;
        pos = succ;
        succ = 2 * pos + 1;
    }

    this->heap[ pos ] = index;
    this->positionVector[ index ] = pos;

    moveUpPos( pos, position );
}


template < typename Item, class IDPolicy >
const typename DynamicPriorityQueue< Item, IDPolicy >::ID
DynamicPriorityQueue< Item, IDPolicy >::push( const Item& item )
{
    const Index index( getSize() );
    
    this->itemVector.push_back( item );
    // index == pos at this time.
    this->heap.push_back( index );
    this->positionVector.push_back( index );

    const ID id( this->pol.push( index ) );

    moveUpPos( index ); 

//    assert( checkConsistency() );

    return id;
}


template < typename Item, class IDPolicy >
void DynamicPriorityQueue< Item, IDPolicy >::popByIndex( const Index index )
{
    // 1. pop the item from the itemVector.
    this->itemVector[ index ] = this->itemVector.back();
    this->itemVector.pop_back();


    // 2.update index<->ID mapping.
    this->pol.pop( index );

    // 3. swap positionVector[ end ] and positionVector[ index ]
    const Index movedPos( this->positionVector.back() );
    const Index removedPos( this->positionVector[ index ] );

    this->positionVector[ index ] = movedPos;
    this->heap[ movedPos ] = index;
    this->positionVector[ this->heap.back() ] = removedPos;
    this->heap[ removedPos ] = this->heap.back();

    // 4. discard the last item
    this->positionVector.pop_back();
    this->heap.pop_back();

    // the heap needs to be rebuilt unless the removed item is the last.
    if ( removedPos < this->heap.size() )
    {
        movePos( removedPos );
    }
}



template < typename Item, class IDPolicy >
void DynamicPriorityQueue< Item, IDPolicy >::replaceTop( const Item& item )
{
    this->itemVector[ this->heap[0] ] = item;
    moveTop();
    
//    assert( checkConsistency() );
}

template < typename Item, class IDPolicy >
void DynamicPriorityQueue< Item, IDPolicy >::
replace( const ID id, const Item& item )
{
    const Index index( this->pol.getIndex( id ) );
    this->itemVector[ index ] = item;
    move( index );
    
//    assert( checkConsistency() );
}


template < typename Item, class IDPolicy >
void DynamicPriorityQueue< Item, IDPolicy >::dump() const
{
    for( Index i( 0 ); i < heap.size(); ++i )
    {
        printf( "heap %d %d %d\n", 
                i, heap[i], this->itemVector[ this->heap[i] ] );
    }
    for( Index i( 0 ); i < positionVector.size(); ++i )
    {
        printf( "pos %d %d\n", 
                i, positionVector[i] );
    }
}


template < typename Item, class IDPolicy >
const bool DynamicPriorityQueue< Item, IDPolicy >::checkConsistency() const
{
    bool result( true );

    // check sizes of data structures.
    result = result && this->itemVector.size() == getSize();
    result = result && this->heap.size() == getSize();
    result = result && this->positionVector.size() == getSize();

    // assert correct mapping between the heap and the positionVector.
    for( Index i( 0 ); i < getSize(); ++i )
    {
        result = result && this->heap[ i ] <= getSize();
        result = result && this->positionVector[ i ] <= getSize();
        result = result && this->heap[ this->positionVector[i] ] == i;
    }

    // assert correct ordering of items in the heap.

    for( Index pos( 0 ); pos < getSize(); ++pos )
    {
        const Item& item( this->itemVector[ this->heap[ pos ] ] );

        const Index succ( pos * 2 + 1 );
        if( succ < getSize() )
        {
            result = result && 
                this->comp( item, this->itemVector[ this->heap[ succ ] ] );

            const Index rightPos( succ + 1 );
            if( rightPos < getSize() )
            {
                result = result &&  
                    this->comp( item, 
                                this->itemVector[ this->heap[ rightPos ] ] );
            }
        }

    }

    result = result && this->pol.checkConsistency( getSize() );

    return result;
}

} // namespace libecs

#endif // __DYNAMICPRIORITYQUEUE_HPP
