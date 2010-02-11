//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2010 Keio University
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
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//    This file is part of E-Cell System.
//
//    Authors: Koichi TAKAHASHI <shafi@e-cell.org>
//             Naota ISHIKAWA <naota@mag.keio.ac.jp>
//             Satya Nanda Vel Arjunan <satya@sfc.keio.ac.jp>
//             Gabor Bereczki <gabor@e-cell.org>
//             Moriyoshi Koizumi <mozo@sfc.keio.ac.jp>
//             Mitsubishi Space Software Co., LTD.
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/*
 *::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 *    $Id$
 :    $Log$
 :    Revision 1.11  2005/11/19 09:23:59  shafi
 :    Kouichi -> Koichi
 :
 :    Revision 1.10  2005/07/13 15:20:06  shafi
 :    support for BSD contributed by Hisham Zarka.
 :    
 :    Revision 1.9  2004/06/17 16:55:30  shafi
 :    copyright updates
 :    
 :    Revision 1.8  2003/08/17 00:15:34  satyanandavel
 :    update for gcc-3.3.1 in mingw
 :    
 :    Revision 1.7  2003/08/09 07:02:59  satyanandavel
 :    correction of a typo
 :    
 :    Revision 1.6  2003/08/08 13:40:07  satyanandavel
 :    Added support for native windows compilation using MinGW
 :    
 :    Revision 1.5  2002/10/11 15:51:32  shafi
 :    rename: Connection -> VariableReference
 :    
 :    Revision 1.4  2002/10/03 08:40:40  shafi
 :    react -> process, REACTANT -> CONNECTION
 :    
 :    Revision 1.3  2002/10/03 08:19:06  shafi
 :    removed korandom, renamings Process -> Process, Variable -> Variable, VariableReference -> VariableReference, Value -> Value, Coefficient -> Coefficient
 :    
 :    Revision 1.2  2002/06/23 14:45:10  shafi
 :    added ProxyPropertySlot. deprecated UpdatePolicy for ConcretePropertySlot. added NEVER_GET_HERE macro.
 :    
 :    Revision 1.1  2002/04/30 11:21:53  shafi
 :    gabor's vvector logger patch + modifications by shafi
 :    
 :    Revision 1.9  2002/01/08 10:23:02  ishikawa
 :    Functions for process path are defined but do not work in this version
 :    
 :    Revision 1.8  2001/10/21 15:27:12  ishikawa
 :    osif_is_dir()
 :    
 :    Revision 1.7  2001/10/15 17:17:18  ishikawa
 :    WIN32 API for free disk space
 :    
 :    Revision 1.6  2001/10/08 10:14:58  ishikawa
 :    WIN32 API for free disk space
 :    
 :    Revision 1.5  2001/08/10 19:10:34  naota
 :    can be compiled using g++ on Linux and Solaris
 :    
 :    Revision 1.4  2001/03/23 18:51:17  naota
 :    comment for credit
 :
 :    Revision 1.3  2001/01/19 21:18:42  naota
 :    Can be comiled g++ on Linux.
 :
 :    Revision 1.2  2001/01/13 01:31:47  naota
 :    Can be compiled by VC, but does not run.
 :
 :    Revision 1.1  2000/12/30 14:57:31  naota
 :    Initial revision
 :
//END_RCS_HEADER
 *::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 */
#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include "osif.h"

#if defined( WIN32 ) && !defined( __CYGWIN__ )
#include <windows.h>
#include "unistd.h"
#include "win32_utils.h"
#else
#include <unistd.h>
#endif

#ifdef _MSC_VER
#define strdup( x ) _strdup( x )
#endif

#include <string.h>

// Support for BSD contributed by Hisham Zarka.


#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif /* HAVE_SYS_PARAM_H */

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif /* HAVE_SYS_STAT_H */

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif /* HAVE_FCNTL_H */

#ifdef HAVE_SYS_STATFS_H
#include <sys/statfs.h>
#endif /* HAVE_SYS_STATFS_H */

#ifdef HAVE_SYS_STATVFS_H
#include <sys/statvfs.h>
#define statfs statvfs
#else
#ifdef HAVE_SYS_MOUNT_H
#include <sys/mount.h>
#endif /* HAVE_SYS_MOUNT_H */
#endif /* HAVE_SYS_STATVFS_H */

#ifdef HAVE_SYS_VFS_H
#include <sys/vfs.h>
#endif

#include <ctype.h>


#define    MAX_RECTOR_PATH 256
static char    *process_path[MAX_RECTOR_PATH];
int        num_process_path;

long osif_get_pid(void)
{
#if defined( WIN32 ) && !defined( __CYGWIN__ )
    return GetCurrentProcessId();
#else
    return (long)getpid();
#endif
}

off_t osif_disk_free(const char *__path)
{
#if defined( WIN32 ) && !defined( __CYGWIN__ )
    ULARGE_INTEGER free_bytes_available;
    ULARGE_INTEGER total_number_of_bytes;
    ULARGE_INTEGER total_number_of_free_bytes;
    char drive[4];

    if (__path[0] != '\0' && __path[1] == ':')
    {
        drive[0] = __path[0];
        drive[1] = ':';
        drive[2] = '/';
        drive[3] = '\0';
    }
    else
    {
        drive[0] = '.';
        drive[1] = '\0';
    }
    if (!GetDiskFreeSpaceEx(
            drive,
            &free_bytes_available,
            &total_number_of_bytes,
            &total_number_of_free_bytes))
    {
        return -1;
    }

    return (off_t)free_bytes_available.QuadPart >> 10;
#else
    struct statfs statfsbuf;

    if ( statfs( __path, &statfsbuf ) )
    {
        return -1;
    }

    return (off_t)(statfsbuf.f_bsize / 512) *
            (statfsbuf.f_bavail / 2);
#endif /* OS dependent part */
}


int osif_mkdir( const char *__name )
{
#if defined( WIN32 ) && !defined( __CYGWIN__ )
    return CreateDirectory( __name, NULL ) ? 0: -1;
#else
    return mkdir( __name, 0755 );
#endif
}

// return a negative number on error
int osif_is_dir(const char *__name)
{
#if defined( WIN32 ) && !defined( __CYGWIN__ )
    DWORD result = GetFileAttributes( __name );
    if ( INVALID_FILE_ATTRIBUTES == result )
    {
        return -1;
    }

    if ( result != (DWORD)-1 &&
         (result & FILE_ATTRIBUTE_DIRECTORY) )
    {
        return 1;
    } else {
        return 0;
    }
#else
    struct stat statbuf;
    if ( stat( __name, &statbuf ) )
    {
        return -1;
    }

     return (S_ISDIR ( statbuf.st_mode)) && !(S_IFDIR & statbuf.st_rdev);
#endif
}

int osif_add_path( const char *dir, int to_first )
{
    if ( osif_is_dir(dir) <= 0 )
    {
        return -1; /* not directory */
    }

    if ( MAX_RECTOR_PATH <= num_process_path )
    {
        return -1; /* to many path */
    }

    if ( to_first )
    {
        if ( num_process_path > 0)
        {
            // Blit is enough
            memmove( &process_path[1], &process_path[0],
                    sizeof( process_path[0] ) *
                    (num_process_path - 1) );
        }

        process_path[0] = strdup(dir);

        return 0;
    }
    else
    {
        process_path[num_process_path] = strdup(dir);
    }

    ++num_process_path;
    return 0;
}
