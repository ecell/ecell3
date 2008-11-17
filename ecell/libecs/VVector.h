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
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//	This file is a part of E-CELL2.
//	Original codes of E-CELL1 core were written by Koichi TAKAHASHI
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
/*
 *::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 *	$Id$
 :	$Log$
 :	Revision 1.21  2005/11/19 09:23:59  shafi
 :	Kouichi -> Koichi
 :
 :	Revision 1.20  2005/04/29 01:38:50  shafi
 :	for gcc4.  more necessary.
 :	
 :	Revision 1.19  2004/07/29 03:52:02  bgabor
 :	Fixing bug [ 994313 ] vvector should close files after read and write.
 :	
 :	Revision 1.18  2004/07/13 18:29:34  shafi
 :	extensive logger code cleanup. remaining things are: replace DataPointVector with boost::multi_array, and understand, reconsider and rename getData( s,e,i )
 :	
 :	Revision 1.17  2004/07/04 10:36:15  bgabor
 :	Code cleanup.
 :	
 :	Revision 1.16  2004/06/14 17:53:15  bgabor
 :	Bugfixing in LogginPolicy
 :	
 :	Revision 1.15  2004/06/10 14:29:11  bgabor
 :	New logger policy API introduced.
 :	Logger policy is now four element long.
 :	First: minimum step between logs (0-no lag )
 :	Second: mimimum interval between logs ( 0-no logs )
 :	Third: policy at end of disk space or available space ( 0- throw ex, 1 overwrite old data )
 :	Fourth: available space for the logger in kbytes ( 0- no limit )
 :	
 :	Revision 1.14  2004/05/29 11:54:51  bgabor
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
 :	Revision 1.13  2004/03/14 19:19:54  satyanandavel
 :	minor fix
 :	
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
#ifndef __VVECTOR_H__
#define	__VVECTOR_H__

#include <vector>
#include <string>

#if !defined(HAVE_SSIZE_T)
typedef int ssize_t;
#endif

#include <string.h>
#include <assert.h>

#ifdef WIN32
#include "fcntl.h"
#include "errno.h"
#include "unistd.h"
#else
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#endif

#include "Exceptions.hpp"

#define OPEN_WHEN_ACCESS

namespace libecs {

const unsigned int VVECTOR_READ_CACHE_SIZE = 2048;
const unsigned int VVECTOR_WRITE_CACHE_SIZE = 2048;
const unsigned int VVECTOR_READ_CACHE_INDEX_SIZE = 2;
const unsigned int VVECTOR_WRITE_CACHE_INDEX_SIZE = 2;

class vvector_full : public std::exception { 
public:
  virtual char const* what() throw()
  {
    return "Total disk space or allocated space is full.\n";
  } 
};

class vvector_write_error : public std::exception { 
public:

  virtual char const* what() throw()
  {
    return "I/O error while attempting to write on disk.\n";
  } 
};

class vvector_read_error : public std::exception { 
public:

  virtual char const* what() throw()
  {
    return "I/O error while attempting to read from disk.\n";
  } 
};


class vvector_init_error : public std::exception { 
public:

  virtual char const* what() throw()
  {
    return "VVector initialization error.\n";
  } 
};

class vvectorbase {
  // types
 public:
  typedef void (*cbfp_t)(); // pointer to call back function
  // private valiables
 private:
  static int _serialNumber;
  static std::string _defaultDirectory;
  static int _directoryPriority;

  static std::vector<std::string> _tmp_name;
  static std::vector<int> _file_desc_read;
  static std::vector<int> _file_desc_write;

  static bool _atexitSet;
  static cbfp_t _cb_full;
  static cbfp_t _cb_error;
  static long _margin;

  // protected variables
 protected:
  int _myNumber;
  std::string _file_name;
  int _fdr,_fdw;
  void unlinkfile();

  // protected methods
  void initBase(char const * const dirname);
  void my_open_to_append();
  void my_open_to_read(off_t offset);
  void my_close_read();
  void my_close_write();

 private:
  vvectorbase( vvectorbase const& );

  // constructor, destructor
 public:
  vvectorbase();
  ~vvectorbase();

  // other public method
  static void setTmpDir(char const * const dirname, int);
  static void removeTmpFile();
  static void setCBFull(cbfp_t __c) { _cb_full = __c; };
  static void cbFull();
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
  bool size_fixed;
  int end_policy;
  size_type max_size;
  size_type start_offset;


  libecs::Real LastTime, diff, lastRead;

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
  void setEndPolicy( int );
  int  getEndPolicy();
  void setMaxSize( size_type aMaxSize );
};


//////////////////////////////////////////////////////////////////////
//      implemetation
//////////////////////////////////////////////////////////////////////


template<class T> vvector<T>::vvector()
{
  size_fixed = false;
  end_policy = 0;
  max_size = 0;
  start_offset = 0;
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
}


template<class T> vvector<T>::~vvector()
{
  // do nothing
}

template<class T> void vvector<T>::setEndPolicy( int anEndPolicy)
{
  end_policy=anEndPolicy;
}

template<class T> int vvector<T>::getEndPolicy()
{
  return end_policy;
}

template<class T> void vvector<T>::setMaxSize( size_type aMaxSize )
{
  if (aMaxSize == 0) {
    max_size = 0;
  }
  else{
    max_size = ((aMaxSize/VVECTOR_WRITE_CACHE_SIZE)+1)*VVECTOR_WRITE_CACHE_SIZE;
  }
}

template<class T> void vvector<T>::push_back(const T & x)
{
  ssize_t red_bytes; 
  bool write_successful;
  if(VVECTOR_WRITE_CACHE_SIZE <= _cacheWNum)
    { 
      throw vvector_write_error();
    }
    
  _cacheWV[_cacheWNum] = x;
  if (_cacheWNum==0)
    {
      _cacheWI[0]=_size;
    }
  _cacheWI[1] = _size;
  if (size_fixed)
    {
      _cacheWI[0]--;
    }
  else
    {
      _size++;
    }
  _cacheWNum++;

  if ( VVECTOR_WRITE_CACHE_SIZE <= _cacheWNum )  
    {
      // first try to append
      if (!size_fixed)
	{
	#ifdef OPEN_WHEN_ACCESS
	   my_open_to_append();
	#endif /*OPEN_WHEN_ACCESS*/
	  red_bytes = write( _fdw, _cacheWV, sizeof(T) * VVECTOR_WRITE_CACHE_SIZE );
	#ifdef OPEN_WHEN_ACCESS
	  my_close_write();
    #endif /*OPEN_WHEN_ACCESS*/

	  write_successful = ( red_bytes == sizeof(T) * VVECTOR_WRITE_CACHE_SIZE );
	  if ( (!write_successful )  || ( _size == max_size ) )
	    {
	      if (end_policy == 0)
		{
		  if (_size>0)
		    {
		      throw vvector_full();
		    }
		  else{
		    throw vvector_write_error();
		  }
		}
	      else 
		{
		  size_fixed=true;
		  if ( !write_successful )
		    {
		      _size-=VVECTOR_WRITE_CACHE_SIZE;
		    }
		    
		}
	    }// if of write statment
	} //if of append condition
	
      if (size_fixed){
	//try to seek start offset
    #ifdef OPEN_WHEN_ACCESS
	  my_open_to_append();
    #endif /*OPEN_WHEN_ACCESS*/

	if (lseek(_fdw, static_cast<off_t>((start_offset) * sizeof(T)), SEEK_SET) == static_cast<off_t>(-1)) 
    {
	  my_close_write();
	    throw vvector_write_error();
	  }

    red_bytes = write(_fdw, _cacheWV, sizeof(T) * VVECTOR_WRITE_CACHE_SIZE);
	#ifdef OPEN_WHEN_ACCESS
	  my_close_write();
    #endif /*OPEN_WHEN_ACCESS*/

	if ( red_bytes != sizeof(T) * VVECTOR_WRITE_CACHE_SIZE) 
	  {
	    throw vvector_write_error();
	  }
	start_offset += VVECTOR_WRITE_CACHE_SIZE;
	if (start_offset >= _size)
	  { 
	    start_offset -= _size;
	  }
	  
	  
      }
      _cacheWNum = 0;
	
    }
}


template<class T>  T const & vvector<T>::operator [] (size_type i)
{
  return at(i);
}


template<class T>  T const & vvector<T>::at(size_type i)
{

  assert(i < _size);
  // read cache only makes sense when not fixed size
  if (!size_fixed){
    if (( i >= _cacheRI[0])&&(_cacheRI[1]>=i)){
      //calculate i's position
      return _cacheRV[i-_cacheRI[0]];
    }
  }

  if ((i>=_cacheWI[0])&&(_cacheWI[1]>=i)){
    //calculate i's position

    return _cacheWV[i-_cacheWI[0]];
  }
  size_type i2=i; //forward sequential read assumed
  size_type log_read_start, phys_read_start;
  size_t half_size,read_interval;
  read_interval=VVECTOR_READ_CACHE_SIZE;
  
  // detect sequential read ( only in case of not fixed read )
  if (!size_fixed){
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
  }
  ssize_t num_red;
  size_type num_to_read = _size - i2 ;
  if (VVECTOR_READ_CACHE_SIZE < num_to_read) {
    num_to_read = VVECTOR_READ_CACHE_SIZE;
  }
  log_read_start=i2;
  if (size_fixed){
    phys_read_start=(i2+start_offset+_cacheWNum);
    if (phys_read_start>=_size){ phys_read_start-=_size;}
  }
  else{
    phys_read_start=i2;
  }
   #ifdef OPEN_WHEN_ACCESS
     my_open_to_read( off_t( phys_read_start * sizeof(T) ));
    #endif /*OPEN_WHEN_ACCESS*/
  num_red = read(_fdr, _cacheRV, num_to_read * sizeof(T));
   #ifdef OPEN_WHEN_ACCESS
  my_close_read();
    #endif /*OPEN_WHEN_ACCESS*/
  if (num_red < 0) {
    throw vvector_read_error();

  }
  num_red /= sizeof(T);
  _cacheRI[0]=log_read_start;
  _cacheRI[1]=log_read_start+num_red-1;

  _buf = _cacheRV[i-log_read_start];
  return _buf;
}


template<class T> void vvector<T>::clear()
{
  unlinkfile();
}

} // namespace libecs

#endif	/* __VVECTOR_H__ */
