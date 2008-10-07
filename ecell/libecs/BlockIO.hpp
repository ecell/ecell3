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

#ifndef __BLOCKIO_H__
#define __BLOCKIO_H__

#include <stddef.h>
#include "Exceptions.hpp"
#include "libecs.hpp"

typedef int fildes_t;

namespace libecs
{

class BlockIO
{
public:
    typedef off_t size_type;
    typedef off_t offset_type;
    typedef ptrdiff_t difference_type;
    typedef int access_flag_type;
    enum lock_type {
        LCK_DEFAULT,
        LCK_READ,
        LCK_READ_WRITE
    };

public:
    static const access_flag_type READ  = 0x0001;
    static const access_flag_type WRITE = 0x0002;

protected:
    int refcount;

protected:
    BlockIO();

public:
    virtual ~BlockIO();
    virtual access_flag_type getAccess() = 0;
    virtual void sync() = 0;
    bool lock();
    virtual bool lock( offset_type offset, size_type size,
                       enum lock_type type = LCK_DEFAULT ) = 0;
    bool unlock();
    virtual bool unlock( offset_type offset, size_type size ) = 0;
    virtual BlockIO* map( offset_type offset, size_type size ) = 0;
    virtual size_type size() const = 0;
    void dispose();
    virtual operator void*() = 0;

    BlockIO* clone();
};

inline BlockIO::BlockIO(): refcount( 1 )
{
}

inline bool
BlockIO::lock()
{
  return lock(0, size());
}

inline bool
BlockIO::unlock()
{
  return unlock(0, size());
}

inline BlockIO*
BlockIO::clone()
{
    return map( 0, size() );
}

inline void BlockIO::dispose()
{
    if (--refcount > 0)
        return;

    delete this;
}

class ConcreteBlockIO;

class FileIO
      : public BlockIO
{
    friend class ConcreteBlockIO;
protected:
    const char* path;
    fildes_t hdl;
    void* ptr;
    access_flag_type access;
    size_type mapped_sz;

public:
    FileIO( fildes_t hdl, char *path, access_flag_type access );
    static FileIO* create( const char *path, access_flag_type access );
    void resize( offset_type size );
    const char *getPath();
    virtual ~FileIO();
    virtual access_flag_type getAccess();
    virtual void sync();
    virtual bool lock( offset_type offset, size_type size,
                       enum lock_type type = LCK_DEFAULT );
    virtual bool unlock( offset_type offset, size_type size );
    virtual BlockIO* map( offset_type offset, size_type size );
    virtual size_type size() const;
    virtual operator void*();
};

inline FileIO::FileIO(fildes_t _hdl, char *_path, access_flag_type _access)
: hdl(_hdl), path(_path), ptr(0), mapped_sz(0), access(_access)
{
}

inline const char* FileIO::getPath()
{
    return path;
}

inline FileIO::access_flag_type FileIO::getAccess()
{
    return access;
}

class ConcreteBlockIO
      : public BlockIO
{
protected:
    FileIO* super;
    void* ptr;
    offset_type offset;
    size_type sz;

private:
    size_t align(offset_type& off, size_type& sz);

public:
    ConcreteBlockIO( FileIO* super, offset_type offset, size_type size );
    virtual ~ConcreteBlockIO();
    virtual access_flag_type getAccess();
    virtual void sync();
    virtual bool lock( offset_type offset, size_type size,
                       enum lock_type type = LCK_DEFAULT );
    virtual bool unlock( offset_type offset, size_type size );
    virtual BlockIO* map( offset_type offset, size_type size );
    virtual size_type size() const;
    virtual operator void*();
};

class VirtualBlockIO
      : public BlockIO
{
protected:
    BlockIO* super;
    offset_type offset;
    size_type sz;

public:
    VirtualBlockIO(BlockIO* super, offset_type offset, size_type size);
    virtual ~VirtualBlockIO();
    virtual access_flag_type getAccess();
    virtual void sync();
    virtual bool lock( offset_type offset, size_type size, 
                       enum lock_type type = LCK_DEFAULT );
    virtual bool unlock( offset_type offset, size_type size );
    virtual BlockIO* map( offset_type offset, size_type size );
    virtual size_type size() const;
    virtual operator void*();
};


inline ConcreteBlockIO::ConcreteBlockIO( FileIO* _super, offset_type _offset,
                                         size_type _size )
    : super( _super ), offset( _offset ), ptr( 0 ), sz( _size )
{
}

inline BlockIO::access_flag_type
ConcreteBlockIO::getAccess()
{
    return super->getAccess();
}

inline void ConcreteBlockIO::sync()
{
    super->sync();
}

inline bool
ConcreteBlockIO::lock( offset_type offset, size_type size,
                       enum lock_type type )
{
    return super->lock( safe_add< off_t >( offset, this->offset ), size, type );
}

inline bool ConcreteBlockIO::unlock(offset_type offset, size_type size)
{
    return super->unlock( safe_add< off_t >( offset, this->offset ), size );
}

inline BlockIO* ConcreteBlockIO::map(offset_type offset, size_type size)
{
    BlockIO* retval;
    if ( safe_add< size_type >( offset, size ) <= size )
    {
        refcount++;
        retval = new VirtualBlockIO( this, offset, size );
    }
    else
    {
        retval = super->map(
            safe_add< off_t >( offset, this->offset ), size );
    }
    return retval;
}

inline BlockIO::size_type ConcreteBlockIO::size() const
{
    return sz;
}

inline VirtualBlockIO::VirtualBlockIO( BlockIO* _super,
                                       offset_type _offset,
                                       size_type _size)
    : super( _super ), offset( _offset ), sz( _size )
{
}

inline VirtualBlockIO::~VirtualBlockIO()
{
    super->dispose();
}

inline void VirtualBlockIO::sync()
{
    super->sync();
}

inline BlockIO*
VirtualBlockIO::map( offset_type offset, size_type size )
{
    BlockIO* retval = new VirtualBlockIO( this, offset, size );
    refcount++;
    return retval;
}

inline int
VirtualBlockIO::getAccess()
{
    return super->getAccess();
}

inline BlockIO::size_type VirtualBlockIO::size() const
{
    return sz;
}

inline bool VirtualBlockIO::lock(offset_type offset, size_type size,
                                 enum lock_type type)
{
    if (offset < 0 || offset >= static_cast<off_t>(this->sz))
    {
        throw std::out_of_range("VirtualBlockIO::lock");
    }

    if (safe_add<size_type>(offset, size) > this->sz)
    {
        throw std::out_of_range("VirtualBlockIO::lock");
    }

    return super->lock(safe_add<offset_type>(this->offset, offset), size, type);
}

inline bool VirtualBlockIO::unlock(offset_type offset, size_type size)
{
    if (offset < 0 || offset >= static_cast< off_t >( this->sz ))
    {
        throw std::out_of_range("VirtualBlockIO::lock");
    }

    if (safe_add< size_type >(offset, size) > this->sz)
    {
        throw std::out_of_range("VirtualBlockIO::lock");
    }

    return super->unlock(safe_add<offset_type>(this->offset, offset), size);
}

inline VirtualBlockIO::operator void*()
{
    return reinterpret_cast< unsigned char * >(
            static_cast<void *>( *super ) ) + offset;
}

template<typename T> class TypedBlock
{
private:
    BlockIO *bio;

public:
    TypedBlock( BlockIO* const& _bio ): bio( _bio ) {}
    ~TypedBlock() {}

    bool lock() const
    {
        return bio->lock(0, sizeof(T));
    }

    bool unlock() const
    {
        return bio->unlock(0, sizeof(T));
    }

    void dispose() const
    {
        bio->dispose();
    }

    void sync() const
    {
        bio->sync();
    }

    BlockIO* channel() const
    {
        return bio;
    }

    operator T&() const
    {
        return *reinterpret_cast<T*>( static_cast<void*>( *bio ) );
    }
};

template<typename T> class TypedArrayBlock
{
public:
    typedef size_t size_type;

private:
    BlockIO* bio;
    size_type block_count;
    BlockIO::size_type block_size;

public:
    TypedArrayBlock( BlockIO* const& _bio, size_t _count )
    : bio( _bio ), block_count( _count ),
      block_size( _count * sizeof( T ) )
    {
        BOOST_ASSERT( block_size <= _bio->size() );
    }

    ~TypedArrayBlock() {}

    bool lock() const
    {
        return bio->lock( 0, block_size );
    }

    bool unlock() const
    {
        return bio->unlock( 0, block_size );
    }

    void dispose() const
    {
        bio->dispose();
    }

    void sync() const
    {
        bio->sync();
    }

    void size() const
    {
        return block_size;
    }

    size_type count() const
    {
        return block_count;
    }

    BlockIO* channel() const
    {
        return bio;
    }

    T& operator[](size_t idx) const
    {
        if ( idx >= block_count )
        {
            throw std::out_of_range("TypedArrayBlock::operator[]");
        }

        T* ptr = reinterpret_cast<T*>(static_cast<void *>(*bio));

        return ptr[idx];
    }

    BlockIO* operator->() const
    {
        return bio;
    }

    TypedArrayBlock<T>& operator=(const TypedArrayBlock<T>& that)
    {
        if ( bio != that.bio )
        {
            bio->sync();
            bio->dispose();
            bio = that.bio;
        }
        block_size = that.block_size;
        block_count = that.block_count;

        return *this;
    }
};

} // namespace libecs

#endif /* __BLOCKIO_H__ */

