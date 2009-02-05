//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2009 Keio University
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
/*
 *::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 *	$Id$
 :	$Log$
 :	Revision 1.16  2005/11/19 09:23:59  shafi
 :	Kouichi -> Koichi
 :
 :	Revision 1.15  2004/07/29 03:52:02  bgabor
 :	Fixing bug [ 994313 ] vvector should close files after read and write.
 :	
 :	Revision 1.14  2004/07/13 18:29:34  shafi
 :	extensive logger code cleanup. remaining things are: replace DataPointVector with boost::multi_array, and understand, reconsider and rename getData( s,e,i )
 :	
 :	Revision 1.13  2004/06/17 16:55:30  shafi
 :	copyright updates
 :	
 :	Revision 1.12  2004/06/04 09:18:28  satyanandavel
 :	bugfix.
 :	----------------------------------------------------------------------
 :	
 :	Revision 1.11  2004/05/29 11:54:51  bgabor
 :	Introducing logger policy .
 :	User cen set and get the policy for a certain logger either when the logger is creater or anytime later.
 :	logger policy is a 3 element list of numbers.
 :	first element:	0-log all
 :			1-log every xth step
 :			2-log at the passing of x secs
 :	second element:	policy at the end
 :			0-throw exception when disk space is used up
 :			1-start overwriting earliest data
 :	third element:	x parameter for minimum step or time interval for first element
 :	
 :	Revision 1.10  2004/01/21 06:26:30  shafi
 :	a fix from gabor, remove tmp files when exitting abnormally
 :	
 :	Revision 1.9  2003/09/27 13:41:07  bgabor
 :	Bugfix.
 :
 :	Revision 1.6  2003/09/27 12:39:15  satyanandavel
 :	more compatibility issues in Windows
 :
 :	Revision 1.5  2003/09/22 04:28:43  bgabor
 :	Fixed a serious undefined reference to my_open_to_read bug in VVector.
 :
 :	Revision 1.4  2003/07/20 06:05:35  bgabor
 :
 :	added support for large files
 :
 :	Revision 1.3  2003/03/18 09:06:36  shafi
 :	logger performance improvement by gabor
 :
 :	Revision 1.2 2003/03/01 02:43:51  shafi
 :	changed envvar name: ECSTMPDIR to VVECTORTMPDIR, changed defult vvector dir: /var/tmp to /tmp
 :
 :	Revision 1.1  2002/04/30 11:21:53  shafi
 :	gabor's vvector logger patch + modifications by shafi
 :
 :	Revision 1.8  2001/10/21 15:27:49  ishikawa
 :	Warn and exit if temprary directory not exist.
 :
 :	Revision 1.7  2001/10/15 17:18:26  ishikawa
 :	improved program interface
 :
 :	Revision 1.6  2001/07/23 23:04:22  naota
 :	Detailed error message.
 :
 :	Revision 1.5  2001/03/23 18:51:17  naota
 :	comment for credit
 :
 :	Revision 1.4  2001/03/08 17:19:55  naota
 :	cp_stop()
 :
 :	Revision 1.3  2001/03/07 18:28:14  naota
 :	default directory is c:/temp on MS-Windows
 :
 :	Revision 1.2  2001/01/19 21:18:42  naota
 :	Can be comiled g++ on Linux.
 :
 :	Revision 1.1  2001/01/10 11:44:46  naota
 :	batch mode running.
 :
 :	Revision 1.1  2000/12/30 14:57:13  naota
 :	Initial revision
 :
 //END_RCS_HEADER
 *::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 */
#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include "VVector.h"
#include "osif.h"

#include <stdlib.h>
#include <string.h>

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif /* HAVE_SYS_STAT_H */

#ifdef _MSC_VER
#define strdup( x ) _strdup( x )
#endif /* _MSC_VER */

#if defined( WIN32 ) && !defined( __CYGWIN__ )
#define PATH_SEPARATOR '\\'
#else
#define PATH_SEPARATOR '/'
#endif

#ifndef O_BINARY
// Must be for UNIX.
#define O_BINARY 0
#endif /* O_BINARY */

#ifndef O_LARGEFILE
// Must be for UNIX.
#define O_LARGEFILE 0
#endif /* O_LARGEFILE */

namespace libecs {

int vvectorbase::_serialNumber = 0;
char const *vvectorbase::_defaultDirectory = NULL;
int vvectorbase::_directoryPriority = 999;
std::vector<char const *> vvectorbase::_tmp_name;
std::vector<int> vvectorbase::_file_desc_read;
std::vector<int> vvectorbase::_file_desc_write;
bool vvectorbase::_atexitSet = false;
vvectorbase::cbfp_t vvectorbase::_cb_error = NULL;
vvectorbase::cbfp_t vvectorbase::_cb_full = NULL;
long vvectorbase::_margin = 10 * 1024; // by K bytes


static void checkDiskFull(char const * const path, int mustCheck)
{
  const long diskFreeMargin =  1024L; // K bytes
  // const int checkInterval = 1024;
  const int checkInterval = 10;
  static int skipCounter = 0;

  if (mustCheck != 0 || (checkInterval <= skipCounter)) {
    skipCounter = 0;
    errno = 0;
    off_t kbytes_free = osif_disk_free(path);
    if (kbytes_free < diskFreeMargin) {
      vvectorbase::cbFull();
    }
  } else {
    skipCounter++;
  }
}

static const char* get_temp_dir()
{
#if defined( WIN32 ) && !defined( __CYGWIN__ )
    return libecs_win32_get_temporary_directory();
#else
    static volatile const char *val = NULL;
    if (val) // should be atomic
        return (const char *)val;

    const char* _val;
    struct stat sb;

    _val = getenv("TMPDIR");
    if ( _val && stat( _val, &sb ) == 0 &&
            ( sb.st_mode & S_IFDIR ) )
    {
        val = _val;
        return _val;
    }

    _val = getenv("TEMPDIR");
    if ( _val && stat( _val, &sb ) == 0 &&
            ( sb.st_mode & S_IFDIR ) )
    {
        val = _val;
        return _val;
    }

    _val = "/var/tmp";
    if ( _val && stat( _val, &sb) == 0 &&
            ( sb.st_mode & S_IFDIR ) )
    {
        val = _val;
        return _val;
    }

    _val = "/tmp";
    if ( _val && stat( _val, &sb) == 0 &&
            ( sb.st_mode & S_IFDIR ) ) {
        val = _val;
        return _val;
    }
#endif
}

vvectorbase::vvectorbase()
{
  if (_defaultDirectory == NULL) {
    char const *envVal = getenv("VVECTORTMPDIR");
    if (envVal != NULL) {
      _defaultDirectory = strdup(envVal);
      _directoryPriority = 3;
    }
    else
    {
	  _defaultDirectory = strdup( get_temp_dir() );
	  _directoryPriority = 4;
    }
  }
  _myNumber = _serialNumber;
  _serialNumber++;
  if (!_atexitSet) {
    _atexitSet = true;
    if (atexit(removeTmpFile) != 0) {
        throw vvector_init_error();
    }
  }

}


vvectorbase::~vvectorbase()
{
  if (0 <= _fdr) {
    close(_fdr);
    _fdr = -1;
  }
  if (0 <= _fdw) {
    close(_fdw);
    _fdw = -1;
  }
  unlinkfile();
}

void vvectorbase::unlinkfile()
{
  if (_file_name) {
    unlink(_file_name); // ignore error
    free(_file_name);
    _file_name = NULL;
  }
}


void vvectorbase::setTmpDir(char const * const dirname, int priority)
{
  assert(dirname != NULL);
  assert(dirname[0] != '\0');
  if (priority < _directoryPriority) {
    _directoryPriority = priority;
    if (_defaultDirectory != NULL) {
      free(const_cast<char*>(_defaultDirectory));
    }
    _defaultDirectory = strdup(dirname);
  }
}


void vvectorbase::removeTmpFile()
{
  std::vector<char const *>::iterator iii;

#ifndef OPEN_WHEN_ACCESS
  std::vector<int>::iterator ii;
  for (ii = _file_desc_read.begin(); ii != _file_desc_read.end(); ii++) {

    if (0 <= *ii) {
      close(*ii);
    }
  }

  for (ii = _file_desc_write.begin(); ii != _file_desc_write.end(); ii++) {

    if (0 <= *ii) {
      close(*ii);
    }
  }
#endif /* OPEN_WHEN_ACCESS */


  for (iii = _tmp_name.begin(); iii != _tmp_name.end(); iii++) {

    unlink (*iii);
  }
}


void vvectorbase::initBase(char const * const dirname)
{
  char pathname[256];
  char filename[256];
  if (dirname != NULL) {
    strcpy(pathname, dirname);
  } else {
    strcpy(pathname, _defaultDirectory);
  }
  if (pathname[strlen(pathname) - 1] != PATH_SEPARATOR) {
    static const char sep[2] = { PATH_SEPARATOR, 0 };
    strcat(pathname, sep);
  }
  if (osif_is_dir(pathname) == 0) {
    throw vvector_init_error();
  }
  checkDiskFull(pathname, 1);
  sprintf(filename, "vvector-%ld-%04d",
	  osif_get_pid(), _myNumber);
  strcat(pathname, filename);
  _file_name = strdup(pathname);
  _tmp_name.push_back(_file_name);

  _fdw = open(_file_name, O_WRONLY | O_CREAT | O_TRUNC | O_BINARY |O_LARGEFILE, 0600);
  _fdr = open(_file_name, O_RDONLY | O_BINARY | O_LARGEFILE );

  if (_fdw < 0) 
    {
        throw vvector_init_error();
    } 

  if (_fdr < 0) 
    {
        throw vvector_init_error();
    }

 #ifndef OPEN_WHEN_ACCESS
  _file_desc_write.push_back(_fdr);
  _file_desc_write.push_back(_fdw);
 #endif /*OPEN_WHEN_ACCESS*/

 #ifdef OPEN_WHEN_ACCES
 close(_fdw);
 _fdw = -1;
 close(_fdr);
 _fdr = -1;
 #endif /*OPEN_WHEN_ACCESS*/
 


}


void vvectorbase::my_open_to_append()
{
  checkDiskFull(_file_name, 0);
  _fdw = open(_file_name, O_WRONLY | O_BINARY |O_LARGEFILE);
  if (_fdw < 0) 
   {
      throw vvector_write_error();
  }
  if ( lseek( _fdw, 0, SEEK_END ) == -1 )
    {
    my_close_write();
       throw vvector_write_error();
    }
}


void vvectorbase::my_open_to_read(off_t offset)
{
  if (_fdr<0) {
    _fdr = open(_file_name, O_RDONLY | O_BINARY|O_LARGEFILE );
  }
  if (_fdr < 0) {
        throw vvector_read_error();

  }
  if (lseek(_fdr, offset, SEEK_SET) == static_cast<off_t>(-1)) {
        throw vvector_read_error();

  }
}

void vvectorbase::my_close_read()
{
  assert (0 <= _fdr);
  if (close(_fdr) < 0) {
        throw vvector_read_error();
  }
  _fdr = -1;
}

void vvectorbase::my_close_write()
{
  assert (0<=_fdw);
  if (close(_fdw) < 0) 
    {
        throw vvector_read_error();
  }

  _fdw = -1;
}


////////////////////////////////////////////////////////////////////////
//	error handlar
////////////////////////////////////////////////////////////////////////
/* static */ void vvectorbase::cbFull()
{
  if (_cb_full != NULL) {
    (*_cb_full)();
  } else {
        throw vvector_full();

  }
}

} // namespace libecs
