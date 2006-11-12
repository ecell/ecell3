/*
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is a part of E-Cell Simulation Environment package
//
//                Copyright (C) 1996-2001 Keio university
//   Copyright (C) 1998-2001 Japan Science and Technology Corporation (JST)
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
 */
/*
 *::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 *	$Id$
 :	$Log$
 :	Revision 1.11  2005/11/19 09:23:59  shafi
 :	Kouichi -> Koichi
 :
 :	Revision 1.10  2005/07/13 15:20:06  shafi
 :	support for BSD contributedby Hisham Zarka.
 :	
 :	Revision 1.9  2004/06/17 16:55:30  shafi
 :	copyright updates
 :	
 :	Revision 1.8  2003/08/17 00:15:34  satyanandavel
 :	update for gcc-3.3.1 in mingw
 :	
 :	Revision 1.7  2003/08/09 07:02:59  satyanandavel
 :	correction of a typo
 :	
 :	Revision 1.6  2003/08/08 13:40:07  satyanandavel
 :	Added support for native windows compilation using MinGW
 :	
 :	Revision 1.5  2002/10/11 15:51:32  shafi
 :	rename: Connection -> VariableReference
 :	
 :	Revision 1.4  2002/10/03 08:40:40  shafi
 :	react -> process, REACTANT -> CONNECTION
 :	
 :	Revision 1.3  2002/10/03 08:19:06  shafi
 :	removed korandom, renamings Process -> Process, Variable -> Variable, VariableReference -> VariableReference, Value -> Value, Coefficient -> Coefficient
 :	
 :	Revision 1.2  2002/06/23 14:45:10  shafi
 :	added ProxyPropertySlot. deprecated UpdatePolicy for ConcretePropertySlot. added NEVER_GET_HERE macro.
 :	
 :	Revision 1.1  2002/04/30 11:21:53  shafi
 :	gabor's vvector logger patch + modifications by shafi
 :	
 :	Revision 1.9  2002/01/08 10:23:02  ishikawa
 :	Functions for process path are defined but do not work in this version
 :	
 :	Revision 1.8  2001/10/21 15:27:12  ishikawa
 :	osif_is_dir()
 :	
 :	Revision 1.7  2001/10/15 17:17:18  ishikawa
 :	WIN32API for free disk space
 :	
 :	Revision 1.6  2001/10/08 10:14:58  ishikawa
 :	WIN32 API for free disk space
 :	
 :	Revision 1.5  2001/08/10 19:10:34  naota
 :	can be compiled using g++ on Linux and Solaris
 :	
 :	Revision 1.4  2001/03/23 18:51:17  naota
 :	comment for credit
 :
 :	Revision 1.3  2001/01/19 21:18:42  naota
 :	Can be comiled g++ on Linux.
 :
 :	Revision 1.2  2001/01/13 01:31:47  naota
 :	Can be compiled by VC, but does not run.
 :
 :	Revision 1.1  2000/12/30 14:57:31  naota
 :	Initial revision
 :
//END_RCS_HEADER
 *::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 */
#include "libecs.hpp"

// Support for BSD contributed by Hisham Zarka.

/* systems that satisfy the below all have <sys/param.h> */
#if (defined(__unix__) || defined(unix)) && !defined(USG)
/* if this is a BSD system, param.h defines the BSD macro */
#include <sys/param.h>
#endif

#if defined(__BORLANDC__) || defined(__WINDOWS__) || defined(__MINGW32__)
/* MS-Windows */
#include <dos.h>
#include <process.h>
#include <wtypes.h>
#include <objidl.h>
#include <winbase.h>
#elif defined(__SVR4)
/* assume SUN Solaris */
#include <unistd.h>
#include <sys/statvfs.h>
#define statfs statvfs
#elif defined(BSD4_4)
/* assume BSD */
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <sys/statvfs.h>
#define statfs statvfs
# else
/* assume Linux */
// #include <process.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#ifndef __APPLE__
#include <sys/vfs.h>
#else
#include <sys/mount.h>
#endif /* __APPLE__ */
#endif /* OS dependent part */
#include <ctype.h>
#include "osif.h"


#define	MAX_RECTOR_PATH 256
static char	*process_path[MAX_RECTOR_PATH];
int		num_process_path;


long		osif_get_pid(void)
{
	return (long)getpid();
}


long		osif_disk_free(const char *__path)
{
#if defined(__BORLANDC__) || defined(__WINDOWS__) || defined(__MINGW32__)
	ULARGE_INTEGER	free_bytes_available;
	ULARGE_INTEGER	total_number_of_bytes;
	ULARGE_INTEGER	total_number_of_free_bytes;
	long		kbfree;
	char		drive[4];

	if (__path[1] == ':') {
		drive[0] = __path[0];
		drive[1] = ':';
		drive[2] = '/';
		drive[3] = '\0';
	} else {
		drive[0] = '.';
		drive[1] = '\0';
	}
	GetDiskFreeSpaceEx(
	  drive,
	  &free_bytes_available,
	  &total_number_of_bytes,
	  &total_number_of_free_bytes);
	kbfree = (long)(free_bytes_available.QuadPart >> 10);
#ifdef	DEBUG_DONE
	printf("%ld k bytes free on %s\n", kbfree, __path);
#endif	/* DEBUG_DONE */
	return kbfree;
#else
	struct statfs statfsbuf;
	errno = 0;
	if (statfs(__path, &statfsbuf)) {
		/* error */
		return 0L;
	}
	return (long)(statfsbuf.f_bsize / 512)
	 * (long)(statfsbuf.f_bavail / 2);
#endif /* OS dependent part */
}


int	osif_mkdir(const char *__name)
{
#if	defined(_MSC_VER)
#include <direct.h>
	return _mkdir(__name);
#elif defined(__BORLANDC__) || defined(__WINDOWS__) || defined(__MINGW32__)
#include <dir.h>
	return mkdir(__name);
#else
	return mkdir(__name, 0755);
#endif
}


int	osif_is_dir(const char *__name)
{
#if defined(__BORLANDC__) || defined(__MINGW32__)
	DWORD result;
	result = GetFileAttributes(__name);
	if (result != (DWORD)-1 && (result & FILE_ATTRIBUTE_DIRECTORY)) {
		return 1;
	} else {
		return 0;
	}
#else
	struct stat statbuf;
	if (stat(__name, &statbuf) ==0
	    && (S_ISDIR ( statbuf.st_mode))) { // gabor
	  //	  && !(S_IFDIR & statbuf.st_rdev)) {
		return 1;
	} else {
		return 0;
	}
#endif
}


int	osif_load_dll(const char *__name)
{
	/* not implemented */
	return -1;
}


int	osif_add_path(const char *dir, int to_first)
{
	if (!osif_is_dir(dir)) {
		return -1; /* not directory */
	} else if (MAX_RECTOR_PATH <= num_process_path) {
		return -1; /* to many path */
	} else if (to_first) {
		int		iii;
		for (iii = num_process_path; iii == 1; iii--) {
			process_path[iii] = process_path[iii - 1];
		}
		process_path[0] = strdup(dir);
		++num_process_path;
		return 0;
	} else {
		process_path[num_process_path] = strdup(dir);
		++num_process_path;
		return 0;
	}
}


/*
 *********************************************************************
 *	for stand alone test
 *********************************************************************
 */
#ifdef TEST
#include <stdio.h>


int	main(int argc, char **argv)
{
	printf("pid = %ld\n", osif_get_pid());
	printf("Disk free by KBytes = %ld on .\n", osif_disk_free("."));
#if defined(__BORLANDC__) || defined(__WINDOWS__) || defined(__MINGW32__)
	printf("Disk free by KBytes = %ld on c:/tmp\n", osif_disk_free("c:/tmp"));
	printf("Disk free by KBytes = %ld on c:/tmp/\n", osif_disk_free("c:/tmp/"));
#endif	/* _Windows */
	printf("osif_is_dir(\"/tmp\") returns %d\n", osif_is_dir("/tmp"));
	printf("osif_is_dir(\"/tmp/\") returns %d\n", osif_is_dir("/tmp/"));
	printf("osif_is_dir(\"/tmp/junk\") returns %d\n", osif_is_dir("/tmp/junk"));
	return 0;
}
#endif /* TEST */
