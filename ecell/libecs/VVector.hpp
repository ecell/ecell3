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


#ifndef __VVECTOR_H__
#define	__VVECTOR_H__

#include <boost/noncopyable.hpp>
#include <boost/assert.hpp>
#include <vector>
#include "libecs.hpp"
#include "Exceptions.hpp"
#include "BlockIO.hpp"

namespace libecs
{

template<typename T, typename BlockIOT = FileIO> class VVector
        : private boost::noncopyable
{
    friend class VVectorMaker;

public:
    template<typename Tcontainer_, typename Tself_>
    class iterator_base
    {
    protected:
        typedef Tself_ Self;

    public:
        typedef Tcontainer_ Container;
        typedef typename Tcontainer_::value_type value_type;
        typedef typename Tcontainer_::difference_type difference_type;
        typedef typename Tcontainer_::size_type size_type;
        typedef ::std::bidirectional_iterator_tag iterator_category;
        typedef typename Tcontainer_::pointer pointer;
        typedef typename Tcontainer_::reference reference;

    public:
        iterator_base(Container& _cntnr, difference_type _cidx)
            : cntnr(_cntnr), cidx(_cidx)
        {
        }

        Self operator+(difference_type diff) const
        {
            Self retval = *this;
            retval.cidx += diff;
            return retval;
        }

        Self operator-(difference_type diff) const
        {
            Self retval = *this;
            retval.cidx -= diff;
            return retval;
        }


        Self operator+=(difference_type diff)
        {
            cidx += diff;
            return *this;
        }

        Self& operator-=(difference_type diff)
        {
            cidx -= diff;
            return *this;
        }

        difference_type operator-(Self const& that) const
        {
            return this.cidx - that.cidx;
        }

        Self& operator++()
        {
            ++cidx;
            return *this;
        }

        Self operator++(int)
        {
            Self save = *this;
            cidx++;
            return save;
        }

        Self& operator--()
        {
            --cidx;
            return *this;
        }

        Self operator--(int)
        {
            Self save = *this;
            cidx--;
            return save;
        }

        bool operator<(const Self& that) const
        {
            return cidx < that.cidx;
        }

        bool operator>(const Self& that) const
        {
            return cidx > that.cidx;
        }

        bool operator==(const Self& that) const
        {
            return cidx == that.cidx;
        }

    protected:
        const Container& cntnr;
        typename Tcontainer_::size_type cidx;
    };

    typedef T value_type;
    typedef ::size_t size_type;
    typedef ::ptrdiff_t difference_type;
    typedef value_type* pointer;
    typedef value_type& reference;
    typedef value_type* const_pointer;
    typedef value_type& const_reference;
    typedef BlockIOT storage_impl_type;

    class iterator: public iterator_base<VVector, iterator>
    {
    protected:
        typedef iterator_base<VVector, iterator> Base;

    public:
        typedef typename VVector::value_type value_type;
        typedef typename VVector::difference_type difference_type;
        typedef typename VVector::reference reference;
        typedef typename VVector::pointer pointer;

    public:
        iterator(typename Base::Container& _cntnr, difference_type _cidx)
            : Base( _cntnr, _cidx )
        {
        }

        iterator( const Base& that )
            : Base( that )
        {
        }

        reference operator*() const
        {
            return (*this)[Base::cidx];
        }

        pointer operator->() const
        {
            return &(*this)[Base::cidx];
        }

        reference operator[](difference_type idx) const
        {
            return Base::cntnr[Base::cidx + idx];
        }
    };

    class const_iterator: public iterator_base<const VVector, const_iterator>
    {
    protected:
        typedef iterator_base<const VVector, const_iterator> Base;

    public:
        typedef typename VVector::value_type value_type;
        typedef typename VVector::difference_type difference_type;
        typedef typename VVector::const_reference reference;
        typedef typename VVector::const_pointer pointer;

    public:
        const_iterator(const typename Base::Container& _cntnr,
                difference_type _cidx)
            : Base( _cntnr, _cidx )
        {
        }

        const_iterator( const Base& that )
            : Base( that )
        {
        }

        reference operator*() const
        {
            return (*this)[Base::cidx];
        }

        pointer operator->() const
        {
            return &(*this)[Base::cidx];
        }

        reference operator[](difference_type idx) const
        {
            return Base::cntnr[Base::cidx + idx];
        }
    };

private:
    struct Buffer
    {
        size_type offset;
        TypedArrayBlock<T> elts;
        int refcount;
        struct Buffer* next;
        struct Buffer* prev;
        struct Buffer* older;
        struct Buffer* newer;

        Buffer( size_type _offset, const TypedArrayBlock<T>& _elts )
            : offset(_offset), elts(_elts), refcount(1)
        {
        }

        ~Buffer()
        {
            elts.dispose();
        }

        void dispose()
        {
            if (--refcount == 0) {
                delete this;
            }
        }

        inline void addRef()
        {
            refcount++;
        }
    };

    struct BufferList
    {
        typedef size_t size_type;

        Buffer* first;
        Buffer* last;
        Buffer* oldest;
        Buffer* newest;
        size_type num_items;

        BufferList()
            : first( 0 ), last( 0 ), oldest( 0 ), newest( 0 ), num_items( 0 )
        {
        }

        void add(Buffer* item)
        {
            item->addRef();
            item->prev = last;
            item->next = 0;

            if (last)
            {
                last->next = item;
            }
            else
            {
                first = item;
            }
            last = item;

            item->older = newest;
            item->newer = 0;
            if (newest)
            {
                newest->newer = item;
            }
            else
            {
                oldest = item;
            }
            newest = item;

            num_items++;
        }

        void insert(Buffer* pos, Buffer* item)
        {
            if (!pos)
            {
                add(item);
                return;
            }

            item->addRef();
            item->next = pos;
            item->prev = pos->prev;
            if (!pos->prev)
            {
                first = pos;
            }
            else
            {
                pos->prev->next = item;
            }
            pos->prev = item;

            item->older = newest;
            item->newer = 0;
            if (newest)
            {
                newest->newer = item;
            }
            else
            {
                oldest = item;
            }
            newest = item;

            num_items++;
        }

        void erase(Buffer* pos)
        {
            if (pos->prev)
            {
                pos->prev->next = pos->next;
            }
            else
            {
                first = pos->next;
            }

            if (pos->next)
            {
                pos->next->prev = pos->prev;
            }
            else
            {
                last = pos->prev;
            }

            if (pos->older)
            {
                pos->older->newer = pos->newer;
            }
            else
            {
                oldest = pos->newer;
            }

            if (pos->newer)
            {
                pos->newer->older = pos->older;
            }
            else
            {
                newest = pos->older;
            }

            num_items--;
            pos->dispose();
        }
    };

    struct Header
    {
        size_type count;
        size_t elem_size;
    };

private:
    BlockIOT* fio;
    TypedBlock<Header> hdr;
    BlockIO* header_block;
    BlockIO::offset_type last_size;
    BufferList buffers;
    size_type max_elts_per_buf;
    size_type min_alloc;
    typename BufferList::size_type max_bufs;

private:
    Buffer* get_buffer( size_type offset, size_type count );
    Buffer* find_buffer( size_type offset, size_type count ) const;

public:
    VVector( BlockIOT* bio, size_type max_elts_per_buf = 16384,
             typename BufferList::size_type max_bufs = 16,
             typename BufferList::size_type min_alloc = -1 );
    ~VVector();

    void push_back( const value_type & x );
    value_type & operator []( size_type i );
    value_type const& operator []( size_type i ) const;
    value_type & at( size_type i );
    value_type const& at( size_type i ) const;
    size_type size() const;
    bool empty() const;
    const_iterator begin() const;
    const_iterator end() const;
    iterator begin();
    iterator end();
    T& front();
    T& back();
    T const& front() const;
    T const& back() const;
    void truncate( size_type len );
    void clear();
    void sync() const;
    BlockIOT* channel() const;
};

template<typename T, typename BlockIOT>
inline VVector<T, BlockIOT>::VVector( BlockIOT* _fio,
                                      size_type _max_elts_per_buf,
                                      typename BufferList::size_type _max_bufs,
                                      typename BufferList::size_type _min_alloc)
    : fio(_fio), max_elts_per_buf(_max_elts_per_buf), max_bufs(_max_bufs),
      last_size(0), hdr(fio->map(0, sizeof(Header)))
{
    min_alloc = _min_alloc == static_cast<size_type>(-1) ?
            getpagesize() / sizeof(T): min_alloc;

    if (static_cast<Header&>(hdr).elem_size != sizeof(T)) {
        hdr.dispose();
        throw std::domain_error( "Wrong file format: storage size for an element is different" );
    }
}

template<typename T, typename BlockIOT>
inline VVector<T, BlockIOT>::~VVector()
{
    for (Buffer *i = buffers.first, *next = 0; i; i = next) {
        next = i->next;
        delete i;
    }

    hdr.dispose();
}

template<typename T, typename BlockIOT>
inline typename VVector<T, BlockIOT>::Buffer*
VVector<T, BlockIOT>::get_buffer( size_type offset, size_type count )
{
    size_type end = offset + count;
    Buffer* nearest = 0;
    Buffer* to_be_deleted = 0;

    if (buffers.last &&
        end <= buffers.last->offset + buffers.last->elts.count())
    {
        for (Buffer* b = buffers.last; b; b = b->prev)
        {
            if ( b->offset >= end )
            {
                nearest = b;
                continue;
            }

            if ( offset < b->offset )
            {
                break;
            }

            if ( b->offset + b->elts.count() >= end )
            {
                return b;
            }

            // expand it
            if ( b->offset + max_elts_per_buf >= end )
            {
                count = max_elts_per_buf;
                offset =  b->offset;
                to_be_deleted = b;
                break;
            }

            break;
        }
    }

    size_type alloc_count = std::max< size_type >( count, min_alloc );

    const BlockIO::offset_type block_off = safe_add< BlockIO::offset_type >(
            sizeof(Header),
            safe_mul< BlockIO::offset_type, size_type >(
                sizeof(T), offset ) );
    const BlockIO::size_type block_size =
            safe_mul< BlockIO::offset_type, size_type >(
                sizeof(T), alloc_count);
    const BlockIO::offset_type required_size = block_off + block_size;

    if ( required_size >= last_size )
    {
        last_size = fio->size();
        if ( required_size >= last_size )
        {
            fio->resize(required_size);
        }
    }

    Buffer* new_buffer = new Buffer(
            offset,
            TypedArrayBlock<T>(
                fio->map( block_off, block_size ),
                alloc_count ) );
    buffers.insert(nearest, new_buffer);
    new_buffer->dispose(); // decrement reference count

    if (to_be_deleted)
    {
        buffers.erase( to_be_deleted );
    }

    if (buffers.num_items > max_bufs)
    {
        buffers.erase( buffers.oldest );
    }

    return new_buffer;
}


template<typename T, typename BlockIOT>
inline typename VVector<T, BlockIOT>::Buffer*
VVector<T, BlockIOT>::find_buffer(size_type offset, size_type count) const
{
    size_type end = offset + count;

    if (end >= static_cast< Header& >(hdr).count)
    {
        return 0;
    }

    for (Buffer* b = buffers.last; b; b = b->prev)
    {
        if ( b->offset >= end )
        {
            continue;
        }

        if ( offset < b->offset )
        {
            break;
        }

        if ( b->offset + b->elts.count() >= end )
        {
            return b;
        }
        break;
    }

    return 0;
}

template<typename T, typename BlockIOT>
inline void VVector<T, BlockIOT>::push_back( const value_type & x )
{
    at( static_cast<Header&>(hdr).count ) = x;
}

template<typename T, typename BlockIOT>
inline T& VVector<T, BlockIOT>::at( size_type idx )
{
    Buffer* b = get_buffer(idx, 1);

    if (!b)
    {
        throw std::out_of_range( "VVector<>::at" );
    }

    if (idx >= static_cast<Header&>(hdr).count)
    {
       static_cast<Header&>(hdr).count = idx + 1;
    } 

    return b->elts[idx - b->offset];
}

template<typename T, typename BlockIOT>
inline const T&
VVector<T, BlockIOT>::at( size_type idx ) const
{
    Buffer* b = find_buffer(idx, 1);

    if (!b) 
    {
        throw std::out_of_range( "VVector<>::at" );
    }

    return b->elts[idx - b->offset];
}

template<typename T, typename BlockIOT>
inline T&
VVector<T, BlockIOT>::operator[]( size_type idx )
{
    return at( idx );
}

template<typename T, typename BlockIOT>
inline const T&
VVector<T, BlockIOT>::operator[]( size_type idx ) const
{
    return at( idx );
}

template<typename T, typename BlockIOT>
inline typename VVector<T, BlockIOT>::size_type
VVector<T, BlockIOT>::size() const
{
    return static_cast<Header&>(hdr).count;
}

template<typename T, typename BlockIOT>
inline bool VVector<T, BlockIOT>::empty() const
{
    return static_cast<Header&>(hdr).count == 0;
}

template<typename T, typename BlockIOT>
inline void VVector<T, BlockIOT>::sync() const
{
    for (Buffer *i = buffers.first; i ;i = i->next)
    {
        i->elts.sync();
    }
    hdr.sync();
}

template<typename T, typename BlockIOT>
typename VVector<T, BlockIOT>::const_iterator VVector<T, BlockIOT>::begin() const
{
    return const_iterator(*this, 0);
}

template<typename T, typename BlockIOT>
typename VVector<T, BlockIOT>::const_iterator VVector<T, BlockIOT>::end() const
{
    return const_iterator(*this, size());
}

template<typename T, typename BlockIOT>
typename VVector<T, BlockIOT>::iterator VVector<T, BlockIOT>::begin()
{
    return iterator(*this, 0);
}

template<typename T, typename BlockIOT>
typename VVector<T, BlockIOT>::iterator VVector<T, BlockIOT>::end()
{
    return iterator(*this, size());
}

template<typename T, typename BlockIOT>
T const& VVector<T, BlockIOT>::front() const
{
    return at(0);
}

template<typename T, typename BlockIOT>
T const& VVector<T, BlockIOT>::back() const
{
    return at(size() - 1);
}

template<typename T, typename BlockIOT>
T& VVector<T, BlockIOT>::front()
{
    return at(0);
}

template<typename T, typename BlockIOT>
T& VVector<T, BlockIOT>::back()
{
    return at(size() - 1);
}

template<typename T, typename BlockIOT>
BlockIOT* VVector<T, BlockIOT>::channel() const
{
    return fio;
}

template<typename T, typename BlockIOT>
inline void VVector<T, BlockIOT>::clear()
{
    truncate(0);
}

template<typename T, typename BlockIOT>
inline void
VVector<T, BlockIOT>::truncate( typename VVector<T, BlockIOT>::size_type len )
{
    Buffer* b;
    Buffer* prev = 0;

    for ( b = buffers.last; b && b->offset >= len; b = prev )
    {
        prev = b->prev;
        buffers.erase( b );
    }
}

class VVectorMaker
{
private:
    String theBaseDirectory;

public:
    LIBECS_API static VVectorMaker& getInstance();

public:
    VVectorMaker(const String& baseDirectory);

    template<typename T_> VVector<T_>* create( fildes_t fd,
                                             char* filename = 0 ) const;
    template<typename T_> VVector<T_>* create() const;
};

inline VVectorMaker::VVectorMaker(const String& baseDirectory)
{
    theBaseDirectory += baseDirectory;
    theBaseDirectory += "/XXXXXXXX";
}

template<typename T_>
VVector<T_>*
VVectorMaker::create( fildes_t fd, char* filename ) const
{
    typedef typename VVector<T_>::Header header_type;

    FileIO* fio = new FileIO( fd, filename, FileIO::READ | FileIO::WRITE );
    fio->resize( sizeof( header_type ) );
    header_type* hdr = static_cast< header_type* >( (void *)*fio );
    hdr->elem_size = sizeof( typename VVector<T_>::value_type );
    hdr->count = 0;

    return new VVector<T_>( fio );
}

template<typename T_>
VVector<T_>* VVectorMaker::create() const
{
    fildes_t fd;
    const String::size_type len = theBaseDirectory.length();
    char *buf = new char[len + 1];
    memcpy(buf, theBaseDirectory.data(), len);
    buf[len] = '\0';

    fd = mkstemp(buf);
    if ( fd < 0 )
    {
        throw IOException( "VVectorMaker::create()",
                           "Cannot create temporary file" );
    }

    try
    {
        return create<T_>( fd, buf );
    }
    catch ( IOException& e )
    {
        close(fd);
        unlink(buf);
        delete[] buf;
        throw e;
    }

    // NEVER GET HERE
};


} // namespace libecs

#endif	/* __VVECTOR_H__ */
