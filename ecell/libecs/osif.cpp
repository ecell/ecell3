/*
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is a part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2001 Keio university
//   Copyright (C) 1998-2001 Japan Science and Technology Corporation (JST)
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
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//	This file is a part of E-CELL2.
//	Original codes of E-CELL1 core were written by Kouichi TAKAHASHI
//	<shafi@e-cell.org>.
//	Some codes of E-CELL2 core are minor changed from E-CELL1
//	by Naota ISHIKAWA <naota@mag.keio.ac.jp>.
//	Other codes of E-CELL2 core and all of E-CELL2 UIMAN are newly
//	written by Naota ISHIKAWA.
//	All codes of E-CELL2 GUI are written by
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
 :	Revision 1.1  2002/04/30 11:21:53  shafi
 :	gabor's vvector logger patch + modifications by shafi
 :
 :	Revision 1.9  2002/01/08 10:23:02  ishikawa
 :	Functions for reactor path are defined but do not work in this version
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
#if defined(_Windows)
/* MS-Windows */
#include <dos.h>
#include <process.h>
#include <objidl.h>
#include <wtypes.h>
#include <winbase.h>
#elif defined(__SVR4)
/* assume SUN Solaris */
#include <unistd.h>
#include <sys/statvfs.h>
#define statfs statvfs
# else
/* assume BSD or Linux */
// #include <process.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <sys/vfs.h>
#endif /* OS dependent part */
#include <ctype.h>
#include "osif.h"


#define	MAX_RECTOR_PATH 256
static char	*reactor_path[MAX_RECTOR_PATH];
int		num_reactor_path;


long		osif_get_pid(void)
{
	return (long)getpid();
}


long		osif_disk_free(const char *__path)
{
#if	defined(_Windows)
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
#elif	defined(__BORLANDC__)
#include <dir.h>
	return mkdir(__name);
#else
	return mkdir(__name, 0755);
#endif
}


int	osif_is_dir(const char *__name)
{
#ifdef	_Windows
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
	 && !(S_IFDIR & statbuf.st_rdev)) {
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
	} else if (MAX_RECTOR_PATH <= num_reactor_path) {
		return -1; /* to many path */
	} else if (to_first) {
		int		iii;
		for (iii = num_reactor_path; iii == 1; iii--) {
			reactor_path[iii] = reactor_path[iii - 1];
		}
		reactor_path[0] = strdup(dir);
		++num_reactor_path;
		return 0;
	} else {
		reactor_path[num_reactor_path] = strdup(dir);
		++num_reactor_path;
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
#ifdef	_Windows
	printf("Disk free by KBytes = %ld on c:/tmp\n", osif_disk_free("c:/tmp"));
	printf("Disk free by KBytes = %ld on c:/tmp/\n", osif_disk_free("c:/tmp/"));
#endif	/* _Windows */
	printf("osif_is_dir(\"/tmp\") returns %d\n", osif_is_dir("/tmp"));
	printf("osif_is_dir(\"/tmp/\") returns %d\n", osif_is_dir("/tmp/"));
	printf("osif_is_dir(\"/tmp/junk\") returns %d\n", osif_is_dir("/tmp/junk"));
	return 0;
}
#endif /* TEST */
