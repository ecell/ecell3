
char const Util_C_rcsid[] = "$Id$";
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
// 		This file is part of Serizawa (E-CELL Core System)
//
//	       written by Kouichi Takahashi  <shafi@sfc.keio.ac.jp>
//
//                              E-CELL Project,
//                          Lab. for Bioinformatics,  
//                             Keio University.
//
//             (see http://www.e-cell.org for details about E-CELL)
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// Serizawa is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// Serizawa is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with Serizawa -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER




#include <string.h>
#include <strstream>
#include <time.h>
#include "util/Util.h"



RandomNumberGenerator* theRandomNumberGenerator = 
new RandomNumberGenerator( 
			  // FIXME: this cast is not good: 
			  // should be reinterpret_cast or something.
			  (Float)(time(NULL)) ,
			  RANDOM_NUMBER_BUFFER_SIZE);

int table_lookup(const string& str,const char** table)
{
  for(int i = 0 ; table[i] != NULL ; ++i)
    if(str == string(table[i]))
      return i;
  return NOMATCH;
}

Float asFloat(const string& str)
{
  istrstream ist(str.c_str());
  Float f;
  ist >> f;
  return f;
}

Int asInt(const string& str)
{
  istrstream ist(str.c_str());
  Int l;
  ist >> l;
  return l;
}

string basenameOf(const string& str, string::size_type maxlength)
{
  string::size_type s = str.rfind('/');
  if (s == string::npos) {
    s = 0;
  } else {
    s++;
  }
  string::size_type e = str.rfind('.');
  if (e == string::npos) {
    e = str.size();
  }
  string::size_type l = e - s;
  if (maxlength != 0 && maxlength < l) {
    l = maxlength;
  }
//  cerr << "basenameOF(\"" << str << "\") = \"" << str.substr(s, l) << "\"\n";
  return str.substr(s, l);
}


#ifdef UTIL_TEST

main()
{
  cerr << decodeTypeName(string("13asddfjijiji9lkjkjkjk8abcdefghij"));

}


#endif /* UTIL_TEST */
