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
 :	Revision 1.12  2004/03/14 19:11:57  satyanandavel
 :	sys/io.h is not available in IBM p650
 :
 :	Revision 1.11  2003/09/28 06:17:23  satyanandavel
 :	MinGW version 3 already defines ssize_t
 :	
 :	Revision 1.10  2003/09/27 13:00:58  bgabor
 :	Windows compatibility fix.
 :	
 :	Revision 1.9  2003/09/27 12:39:15  satyanandavel
 :	more compatibility issues in Windows
 :	
 :	Revision 1.8  2003/09/22 04:28:44  bgabor
 :	Fixed a serious undefined reference to my_open_to_read bug in VVector.
 :	
 :	Revision 1.7  2003/08/08 13:13:09  satyanandavel
 :	Added support for MinGW to define type of ssize_t
 :	
 :	Revision 1.6  2003/07/20 06:06:06  bgabor
 :	
 :	Added support for large files.
 :	
 :	Revision 1.5  2003/04/02 11:42:18  shafi
 :	my_open_to_read( off_t )
 :	
 :	Revision 1.4  2003/03/18 09:06:36  shafi
 :	logger performance improvement by gabor
 :	
 :	Revision 1.2  2003/02/03 15:31:56  shafi
 :	changed vvector cache sizes to 1024
 :	
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
#include "ecell/config.h"
#include <sys/types.h>
#if	defined(__GNUC__)
#include <vector>
#else
#include <vector.h>
#endif	/* compiler dependent part */


#if defined(__BORLANDC__)
typedef int ssize_t;
#endif /* __BORLANDC__ */


#include <string.h>
#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include <errno.h>
#if 	defined(__BORLANDC__)
#include <io.h>
#elif	defined(__linux__)
#include <unistd.h>
#if 	!defined(__powerpc__)
#include <sys/io.h>
#endif
#else
#include <unistd.h>
#endif 



const unsigned int VVECTOR_READ_CACHE_SIZE = 2048;
const unsigned int VVECTOR_WRITE_CACHE_SIZE = 2048;
const unsigned int VVECTOR_READ_CACHE_INDEX_SIZE = 2;
const unsigned int VVECTOR_WRITE_CACHE_INDEX_SIZE = 2;

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
  static std::vector<int> _file_desc_read;
  static std::vector<int> _file_desc_write;

  static bool _atexitSet;
  static cbfp_t _cb_full;
  static cbfp_t _cb_error;
  static long _margin;

// protected variables
protected:
  int _myNumber;
  char *_file_name;
  int _fdr,_fdw;
  void unlinkfile();

// protected methods
  void initBase(char const * const dirname);
  void my_open_to_append();
  void my_open_to_read(off_t offset);
  void my_close();

// constructor, destructor
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
  size_type _cacheRI[VVECTOR_READ_CACHE_INDEX_SIZE];
  value_type _cacheWV[VVECTOR_WRITE_CACHE_SIZE];
  size_type _cacheWI[VVECTOR_WRITE_CACHE_INDEX_SIZE];
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
//  void set_direct_read_stats(size_type distance,size_type num_of_elements);
//  void set_direct_read_stats();
};


//////////////////////////////////////////////////////////////////////
//      implemetation
//////////////////////////////////////////////////////////////////////


template<class T> vvector<T>::vvector()
{
  initBase(NULL);
  int counter = VVECTOR_READ_CACHE_INDEX_SIZE;
  do {
    _cacheRI[counter] = -1;
  } while (0 <= --counter);
  counter = VVECTOR_WRITE_CACHE_INDEX_SIZE;
  do {
    _cacheWI[counter] = -1;
  } while (0 <= --counter);
  _cacheWNum = 0;
  _size = 0;
//  direct_read_flag=false;
}


template<class T> vvector<T>::~vvector()
{
  // do nothing
}


template<class T> void vvector<T>::push_back(const T & x)
{
  _cacheWV[_cacheWNum] = x;
  if (_cacheWNum==0){_cacheWI[0]=_size;}
  _cacheWI[1] = _size;
  if (VVECTOR_WRITE_CACHE_SIZE <= ++_cacheWNum) {

    if (write(_fdw, _cacheWV, sizeof(T) * VVECTOR_WRITE_CACHE_SIZE)
	!= sizeof(T) * VVECTOR_WRITE_CACHE_SIZE) {
      fprintf(stderr, "write() failed in VVector.%ld\n", _size);
      cbError();
    }
    _cacheWNum = 0;
//    my_close();
  }
  _size++;
}


template<class T>  T const & vvector<T>::operator [] (size_type i)
{
  assert(i < _size);
  if (( i >= _cacheRI[0])&&(_cacheRI[1]>=i)){
  //calculate i's position
    return _cacheRV[i-_cacheRI[0]];
  }
  if ((i>=_cacheWI[0])&&(_cacheWI[1]>=i)){
  //calculate i's position
    return _cacheWV[i-_cacheWI[0]];
  }
  size_type i2=i; //forward sequential read assumed
 
  size_t half_size,read_interval;
  read_interval=VVECTOR_READ_CACHE_SIZE;
  
  // detect sequential read
  if ((i+1)==_cacheRI[0])
    {
       if (_cacheRI[0]>=read_interval)
          {
            i2=_cacheRI[0]-read_interval;
          }
       else 
          {
            i2=0; 
          }
    }
    else if((_cacheRI[1]+1)!=i){ //not forward sequential, therefore random access
    half_size=read_interval/2;
    if (i>half_size){i2=(i-half_size);} else {i2=0;}
    }

  ssize_t num_red;
  size_type num_to_read = _size - i2 ;
  if (VVECTOR_READ_CACHE_SIZE < num_to_read) {
    num_to_read = VVECTOR_READ_CACHE_SIZE;
  }

  my_open_to_read(static_cast<off_t>((i2) * sizeof(T)));
  num_red = read(_fdr, _cacheRV, num_to_read * sizeof(T));

  if (num_red < 0) {
    fprintf(stderr, "read() failed in VVector. i=%ld, _size=%ld, n=%ld\n",
	    (long)i2, (long)_size, (long)num_to_read);
    cbError();
  }
  num_red /= sizeof(T);

  _cacheRI[0]=i2;
  _cacheRI[1]=i2+num_red-1;
  

  _buf = _cacheRV[i-_cacheRI[0]];

  return _buf;
}


template<class T>  T const & vvector<T>::at(size_type i)
{
  assert(i < _size);
/*  int counter = VVECTOR_READ_CACHE_SIZE - 1;
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
  } while (0 <= --counter);*/
  //check whether i is in range of _cacheRI[0],_cacheRI[1]
  if ((i>=_cacheRI[0])&&(_cacheRI[1]>=i)){
  //calculate i's position
    return _cacheRV[i-_cacheRI[0]];
  }
  if ((i>=_cacheWI[0])&&(_cacheWI[1]>=i)){
  //calculate i's position
    return _cacheWV[i-_cacheWI[0]];
  }
  
  my_open_to_read(static_cast<off_t>((i) * sizeof(T)));
  size_type num_to_read = _size - i;
  if (VVECTOR_READ_CACHE_SIZE < num_to_read) {
    num_to_read = VVECTOR_READ_CACHE_SIZE;
  }
  ssize_t num_red = read(_fdr, _cacheRV, num_to_read * sizeof(T));
  if (num_red < 0) {
    fprintf(stderr, "read() failed in VVector. i=%ld, _size=%ld, n=%ld\n",
	    (long)i, (long)_size, (long)num_to_read);
    cbError();
  }
  num_red /= sizeof(T);
/*  for (ssize_t tmp_index = 0; tmp_index < num_red; tmp_index++) {
    _cacheRI[tmp_index] = i + tmp_index;
  }*/
  _cacheRI[0]=i;
  _cacheRI[1]=i+num_red-1;
  
  _buf = _cacheRV[0];
//  my_close();
  return _buf;
}


template<class T> void vvector<T>::clear()
{
    unlinkfile();
}


#endif	/* __VVECOTOR_H__ */
