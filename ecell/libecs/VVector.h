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
 :	Revision 1.6  2001/10/15 17:18:26  ishikawa
 :	improved program interface
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
 :	Revision 1.2  2001/01/10 11:44:46  naota
 :	batch mode running.
 :
 :	Revision 1.1  2000/12/30 15:09:46  naota
 :	Initial revision
 :
//END_RCS_HEADER
 *::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 */
#ifndef __VVECOTOR_H__
#define	__VVECOTOR_H__
#include <sys/types.h>
#if	defined(__GNUC__)
#include <vector>
#else
#include <vector.h>
#endif	/* compiler dependent part */


#ifdef __BORLANDC__
typedef int ssize_t;
#endif /* __BORLANDC__ */
const unsigned int VVECTOR_READ_CACHE_SIZE = 40;
const unsigned int VVECTOR_WRITE_CACHE_SIZE = 40;


class vvectorbase {
// types
public:
  typedef void (*cbfp_t)(); // pointer to call back function
// private valiables
private:
  static int _serialNumber;
  static char const *_defaultDirectory;
  static int _directoryPriority;

  static std::vector<char const *> _tmp_name;
  static bool _atexitSet;
  static cbfp_t _cb_full;
  static cbfp_t _cb_error;
  static long _margin;

// protected variables
protected:
  int _myNumber;
  char *_file_name;
  int _fd;

// protected methods
  void initBase(char const * const dirname);
  void my_open_to_append();
  void my_open_to_read(long offset);
  void my_close();

// constcuctor, destcuctor
public:
  vvectorbase();
  ~vvectorbase();

// other public method
  static void setTmpDir(char const * const dirname, int);
  static void removeTmpFile();
  static void setCBFull(cbfp_t __c) { _cb_full = __c; };
  static void setCBError(cbfp_t __c) { _cb_error = __c; };
  static void cbFull();
  static void cbError();
  static void margin(long __m) { _margin = __m; }; // by K bytes
};


template<class T> class vvector : public vvectorbase {
// types
public:
  typedef T value_type;
  typedef size_t size_type;

// private valiables
private:
  size_type _size;
  value_type _buf;
  value_type _cacheRV[VVECTOR_READ_CACHE_SIZE];
  size_type _cacheRI[VVECTOR_READ_CACHE_SIZE];
  value_type _cacheWV[VVECTOR_WRITE_CACHE_SIZE];
  size_type _cacheWI[VVECTOR_WRITE_CACHE_SIZE];
  size_type _cacheWNum;

// constructor, destructor
public:
  vvector();
  ~vvector();

// other public methods
public:
  void push_back(const value_type & x);
  value_type const & operator [] (size_type i);
  value_type const & at(size_type i);
  size_type size() const { return _size; };
  void clear();
  static void setDiskFullCB(void(*)());
};


//////////////////////////////////////////////////////////////////////
//      implemetation
//////////////////////////////////////////////////////////////////////
#include <string.h>
#include <memory.h>
#include <stdio.h>
#include <assert.h>
#if 	defined(__BORLANDC__)
#include <io.h>
#elif	defined(__linux__)
#include <unistd.h>
#include <sys/io.h>
#else
#include <unistd.h>
#endif


template<class T> vvector<T>::vvector()
{
  initBase(NULL);
  int counter = VVECTOR_READ_CACHE_SIZE;
  do {
    _cacheRI[counter] = -1;
  } while (0 <= --counter);
  counter = VVECTOR_WRITE_CACHE_SIZE;
  do {
    _cacheWI[counter] = -1;
  } while (0 <= --counter);
  _cacheWNum = 0;
  _size = 0;
}


template<class T> vvector<T>::~vvector()
{
  // do nothing
}


template<class T> void vvector<T>::push_back(const T & x)
{
  _cacheWV[_cacheWNum] = x;
  _cacheWI[_cacheWNum] = _size;
  if (VVECTOR_WRITE_CACHE_SIZE <= ++_cacheWNum) {
    my_open_to_append();
    if (write(_fd, _cacheWV, sizeof(T) * VVECTOR_WRITE_CACHE_SIZE)
	!= sizeof(T) * VVECTOR_WRITE_CACHE_SIZE) {
      fprintf(stderr, "write() failed in VVector.\n");
      cbError();
    }
    _cacheWNum = 0;
    my_close();
  }
  _size++;
}


template<class T>  T const & vvector<T>::operator [] (size_type i)
{
  assert(i < _size);
  int counter = VVECTOR_READ_CACHE_SIZE - 1;
  do {
    if (_cacheRI[counter] == i) {
      return _cacheRV[counter];
    }
  } while (0 <= --counter);
  counter = VVECTOR_WRITE_CACHE_SIZE - 1;
  do {
    if (_cacheWI[counter] == i) {
      return _cacheWV[counter];
    }
  } while (0 <= --counter);
  my_open_to_read(static_cast<off_t>(i * sizeof(T)));
  size_type num_to_read = _size - i;
  if (VVECTOR_READ_CACHE_SIZE < num_to_read) {
    num_to_read = VVECTOR_READ_CACHE_SIZE;
  }
  ssize_t num_red = read(_fd, _cacheRV, num_to_read * sizeof(T));
  if (num_red < 0) {
    fprintf(stderr, "read() failed in VVector. i=%ld, _size=%ld, n=%ld\n",
	    (long)i, (long)_size, (long)num_to_read);
    cbError();
  }
  num_red /= sizeof(T);
  for (ssize_t tmp_index = 0; tmp_index < num_red; tmp_index++) {
    _cacheRI[tmp_index] = i + tmp_index;
  }
  _buf = _cacheRV[0];
  my_close();
  return _buf;
}


template<class T>  T const & vvector<T>::at(size_type i)
{
  assert(i < _size);
  int counter = VVECTOR_READ_CACHE_SIZE - 1;
  do {
    if (_cacheRI[counter] == i) {
      return _cacheRV[counter];
    }
  } while (0 <= --counter);
  counter = VVECTOR_WRITE_CACHE_SIZE - 1;
  do {
    if (_cacheWI[counter] == i) {
      return _cacheWV[counter];
    }
  } while (0 <= --counter);
  my_open_to_read(static_cast<off_t>(i * sizeof(T)));
  size_type num_to_read = _size - i;
  if (VVECTOR_READ_CACHE_SIZE < num_to_read) {
    num_to_read = VVECTOR_READ_CACHE_SIZE;
  }
  ssize_t num_red = read(_fd, _cacheRV, num_to_read * sizeof(T));
  if (num_red < 0) {
    fprintf(stderr, "read() failed in VVector. i=%ld, _size=%ld, n=%ld\n",
	    (long)i, (long)_size, (long)num_to_read);
    cbError();
  }
  num_red /= sizeof(T);
  for (ssize_t tmp_index = 0; tmp_index < num_red; tmp_index++) {
    _cacheRI[tmp_index] = i + tmp_index;
  }
  _buf = _cacheRV[0];
  my_close();
  return _buf;
}


template<class T> void vvector<T>::clear()
{
  if (unlink(_file_name) != 0) {
    fprintf(stderr, "unlink() failed in VVector.\n");
  }
}


#endif	/* __VVECOTOR_H__ */
