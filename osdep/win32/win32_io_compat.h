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
//
// written by Moriyoshi Koizumi <mozo@sfc.keio.ac.jp>
//

#ifndef __WIN32_IO_COMPAT_H
#define __WIN32_IO_COMPAT_H

#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define O_RDONLY    0x0001
#define O_WRONLY    0x0002
#define O_RDWR      0x0003
#define O_CREAT     0x0004
#define O_EXCL      0x0008
#define O_TRUNC     0x0010
#define O_APPEND    0x0020
#define O_DIRECTORY 0x0040 
#define O_NOCTTY    0x0100
#define O_ASYNC     0x0200
#define O_DIRECT    0x0400
#define O_LARGEFILE 0x0800
#define O_NOATIME   0x1000
#define O_NOFOLLOW  0x2000
#define O_NONBLOCK  0x4000
#define O_SYNC      0x8000
#define O_NDELAY    O_NONBLOCK

#define S_IRWXU     00700 
#define S_IRUSR     00400
#define S_IWUSR     00200 
#define S_IXUSR     00100
#define S_IRWXG     00070
#define S_IRGRP     00040
#define S_IWGRP     00020
#define S_IXGRP     00010
#define S_IRWXO     00007
#define S_IROTH     00004
#define S_IWOTH     00002
#define S_IXOTH     00001

#define F_DUPFD     1
#define F_GETFD     2
#define F_SETFD     3
#define F_GETFL     4
#define F_SETFL     5
#define F_GETLK     8
#define F_SETLK     9
#define F_SETLKW   10

#define LOCK_SH     0x01
#define LOCK_EX     0x02
#define LOCK_NB     0x04
#define LOCK_UN     0x08

#define F_RDLCK     1
#define F_UNLCK     2
#define F_WRLCK     3

typedef __int64 off_t;
typedef int mode_t;
typedef unsigned int dev_t;
typedef unsigned __int64 ino_t;

struct flock {
    off_t l_start;
    off_t l_len;
    int   l_pid;
    short l_type;
    short l_whence;
};

struct timespec {
    long tv_sec;
    long tv_nsec;
};

struct stat {
    dev_t st_dev;
    ino_t st_ino;
    time_t st_atime;
    struct timespec st_atimespec;
    time_t st_mtime;
    struct timespec st_mtimespec;
    time_t st_ctime;
    struct timespec st_ctimespec;
    off_t  st_size;
};

extern int libecs_win32_io_open(const char *pathname, int flags, ...);
extern int libecs_win32_io_creat(const char *pathname, mode_t mode);
extern int libecs_win32_io_close(int fd);
extern int libecs_win32_io_read(int fd, void *buf, size_t size);
extern int libecs_win32_io_write(int fd, const void *buf, size_t size);
extern int libecs_win32_io_cntl(int fd, int cmd, ...);
extern off_t libecs_win32_io_seek(int fd, off_t offset, int whence);
extern int libecs_win32_io_truncate(int fd, off_t length);
extern int libecs_win32_io_mkstemps(char *tpl, size_t slen);
extern int libecs_win32_io_stat(int fd, struct stat *sb);
extern int libecs_win32_io_unlink(const char *pathname);

#ifdef __cplusplus
} // extern "C"
#endif


#endif /* __WIN32_IO_COMPAT_H */
