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
// written by Moriyoshi Koizumi <mozo@sfc.keio.ac.jp>
//

#ifndef _LIBECS_WIN32_MMAN_COMPAT_H
#define _LIBECS_WIN32_MMAN_COMPAT_H

#include "stdlib.h"
#include "win32_io_compat.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PROT_NONE   0x00000000
#define PROT_EXEC   0x00000001
#define PROT_WRITE  0x00000002
#define PROT_READ   0x00000004
#define PROT_MASK   (PROT_EXEC | PROT_WRITE | PROT_READ)

#define MAP_EXECUTABLE 0x00000002
#define MAP_PRIVATE    0x00000008
#define MAP_SHARED     0x00000010
#define MAP_FIXED      0x00000020
#define MAP_DENYWRITE  0x00000040
#define MAP_NORESERVE  0x00000080
#define MAP_LOCKED     0x00000100
#define MAP_GROWSDOWN  0x00000200
#define MAP_ANONYMOUS  0x80000000
#define MAP_FILE       0x00000000 
#define MAP_ANON       MAP_ANONYMOUS
#define MAP_32BIT      0x40000000
#define MAP_POPULATE   0x00000400
#define MAP_NONBLOCK   0x00000800

#define MAP_FAILED ((void *)-1)

extern void *libecs_win32_mman_map(void *start, size_t size, int protection,
        int flags, int fd, off_t offset);

extern int libecs_win32_mman_unmap(void *start, size_t size);

extern int libecs_win32_mman_sync(void *start, size_t size, int flags);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* _LIBECS_WIN32_DL_COMPAT_H */
