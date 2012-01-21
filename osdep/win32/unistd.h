//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2012 Keio University
//       Copyright (C) 2005-2009 The Molecular Sciences Institute
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
#ifndef __LIBECS_WIN32_UNISTD_H
#define __LIBECS_WIN32_UNISTD_H

#include "win32_io_compat.h"
#include "win32_utils.h"

#define open libecs_win32_io_open
#define creat libecs_win32_io_creat
#define close(fd) libecs_win32_io_close(fd)
#define read libecs_win32_io_read
#define write libecs_win32_io_write
#define fcntl libecs_win32_io_cntl
#define lseek libecs_win32_io_seek
#define mkstemps libecs_win32_io_mkstemps
#define mkstemp(path) libecs_win32_io_mkstemps(path, 0)
#define ftruncate libecs_win32_io_truncate
#define fstat(fd, sb) libecs_win32_io_stat(fd, sb)
#define unlink(path) libecs_win32_io_unlink(path)
#define getpagesize() libecs_win32_get_page_size()

#endif /* __LIBECS_WIN32_UNISTD_H */
