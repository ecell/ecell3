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
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//	This file is a part of E-Cell2.
//	Original codes of E-Cell1 core were written by Koichi TAKAHASHI
//	<shafi@e-cell.org>.
//	Some codes of E-Cell2 core are minor changed from E-Cell1
//	by Naota ISHIKAWA <naota@mag.keio.ac.jp>.
//	Other codes of E-Cell2 core and all of E-Cell2 UIMAN are newly
//	written by Naota ISHIKAWA.
//	All codes of E-Cell2 GUI are written by
//	Mitsui Knowledge Industry Co., Ltd. <http://bio.mki.co.jp/>
//
//	Latest version is availabe on <http://bioinformatics.org/>
//	and/or <http://www.e-cell.org/>.
//END_V2_HEADER
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#ifndef WIN32
#include <unistd.h>

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#ifdef HAVE_MEMORY_H
#include <memory.h>
#endif

#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif

#ifndef __libecs_errno
#include <errno.h>
#define __libecs_errno errno
#endif

#else

#include "unistd.h"
#include "mman.h"
#include "errno.h"

#endif

#include <string.h>
#include <stdlib.h>

#include "BlockIO.hpp"

#ifdef DEBUG
#include <iostream>
#endif

namespace libecs {

static inline int convert_to_protection_value(BlockIO::access_flag_type flag)
{
    int retval = 0;

    if ((flag & BlockIO::READ))
        retval |= PROT_READ;
    if ((flag & BlockIO::WRITE))
        retval |= PROT_WRITE;

    return retval;
}

BlockIO::~BlockIO()
{
}

FileIO::~FileIO()
{
    if (ptr) {
        munmap(ptr, safe_int_cast<size_t>(size()));
    }    

    if (hdl >= 0) {
        close(hdl);
        hdl = -1;
    }

    delete[] path;
}

FileIO* FileIO::create(const char* path, access_flag_type access)
{
    int acc_val = 0;
    int hdl = -1;

    if (access & READ) {
        if (access & WRITE) {
            acc_val = O_RDWR;
        } else {
            acc_val = O_RDONLY;
        }
    } else {
        if (access & WRITE) {
            acc_val = O_WRONLY;
        }
    }

    acc_val |= O_CREAT;

    hdl = open(path, acc_val, 0600);
    if (hdl < 0) {
        throw IOException(
                "FileIO::openFile()",
                "File could not be opened");
    }

    size_t path_len = strlen(path);
    char* _path = new char[path_len + 1];
    memcpy(_path, path, path_len + 1);

    try
    {
        return new FileIO(hdl, _path, access);
    }
    catch (std::exception& e)
    {
        delete _path;
        close(hdl);
        throw e;
    }

    return 0; // never reached
}

void FileIO::sync()
{
    if (hdl != -1 && ptr) {
        if (msync(ptr, safe_int_cast<size_t>(size()), 0)) {
            throw IOException(
                    "FileIO::sync",
                    "Failed");
        }
    }
}

bool FileIO::lock(size_type offset, size_type size, enum lock_type type)
{
    BOOST_ASSERT(offset >= 0);
    BOOST_ASSERT(size >= 0);

    if (type == LCK_DEFAULT) {
        if (access & WRITE) {
            type = LCK_READ_WRITE;
        } else if (access & READ) {
            type = LCK_READ;
        }
    }

    struct flock ls;
    ls.l_type  = SEEK_SET;
    ls.l_start = offset;
    ls.l_len   = size;
    ls.l_type  = type == LCK_READ ? F_RDLCK: F_WRLCK;

    if (fcntl(hdl, F_SETLKW, ls) < 0) {
        throw IOException(
                "FileIO::lock",
                "Failed");
    }

    return true;
}

bool FileIO::unlock(offset_type offset, size_type size)
{
    BOOST_ASSERT(offset >= 0);
    BOOST_ASSERT(size >= 0);

    if (hdl == -1)
        throw IllegalOperation("FileIO::unlock", "File is not opened");

    struct flock ls;
    ls.l_type  = SEEK_SET;
    ls.l_start = offset;
    ls.l_len   = size;
    ls.l_type  = F_UNLCK;

    if (fcntl(hdl, F_SETLKW, ls) < 0) {
        throw IOException(
                "FileIO::unlock",
                "Failed");
    }

    return true;
}

BlockIO* FileIO::map(offset_type offset, size_type size)
{
    BOOST_ASSERT(size >= 0);
    size_type sz = this->size();

    if (offset < 0 || offset > static_cast<offset_type>(sz)) 
        throw ValueError("FileIO::map", "offset is out of range");

    if (safe_add<size_type>(offset, size) > sz)
        throw ValueError("FileIO::map", "offset + size is out of range");

    refcount++;

    if (!ptr || offset >= mapped_sz)
        return new ConcreteBlockIO(this, offset, size);
    else
        return new VirtualBlockIO(this, offset, size);
}

BlockIO::size_type FileIO::size()
{
    struct stat sb;

    if (fstat(hdl, &sb)) {
        throw IOException(
                "FileIO::size",
                "Failed to retrieve file size");
    }

    return sb.st_size;
}

void FileIO::resize(offset_type size)
{
    if (ftruncate(hdl, size)) {
        throw IOException(
                "FileIO::set_size",
                "File to resize file");
    }
}

void FileIO::dispose()
{
    if (--refcount > 0)
        return;
}

FileIO::operator void*()
{
    size_type cur_sz = size();
    if (this->ptr) {
        if (this->mapped_sz == cur_sz) {
            return ptr;
        }

        if (munmap(this->ptr, safe_int_cast<size_t>(mapped_sz))) {
            throw IOException(
            "FileIO::operator void*",
            "Failed to unmap the previously allocated region");
        }
    }

    void* ptr = mmap(NULL, safe_int_cast<size_t>(cur_sz),
            convert_to_protection_value(access),
            MAP_SHARED, hdl, 0);
    if (MAP_FAILED == ptr) {
        throw IOException(
                "FileIO::operator void*",
                "Failed to map a new region");
    }

    this->ptr = ptr;
    this->mapped_sz = cur_sz;

    return ptr;
}

ConcreteBlockIO::~ConcreteBlockIO()
{
    if (ptr) {
        offset_type offset;
        size_type sz;
        const size_t diff = align(offset, sz);

        munmap(static_cast<char *>(ptr) - diff, safe_int_cast<size_t>(sz));
    }

    super->dispose();
}

inline size_t ConcreteBlockIO::align(offset_type& off, size_type& sz)
{
    const size_t page_sz = getpagesize();
    const size_t diff = safe_int_cast<size_t>(
            this->offset - ( this->offset / page_sz ) * page_sz);
    off = this->offset - diff;
    sz = (size() + diff + page_sz - 1) / page_sz * page_sz;
    return diff;
}

ConcreteBlockIO::operator void*()
{
    if (!ptr) {
        offset_type offset;
        size_type sz;
        const size_t diff = align(offset, sz);

        void* ptr = mmap(NULL, safe_int_cast<size_t>(sz),
            convert_to_protection_value(getAccess()),
            MAP_SHARED, super->hdl, offset);
        if (MAP_FAILED == ptr) {
            throw IOException(
                    "FileIO::operator void*",
                    "Failed");
        }

        this->ptr = static_cast<char *>(ptr) + diff;
    }

    return ptr;
}

} // namespace libecs

