
char const StringList_C_rcsid[] = "$Id$";
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




#include "StringList.h"



StringList::StringList(const string& str, const string& delim,
		       const string& spacer)
{
  parse(str,delim,spacer);
}

void StringList::parse(const string& str,const string& delim,
		       const string& spacer)
{
  if(str == "")
    return;
  string::size_type   s = str.find_first_not_of(spacer);
  string::size_type   e = str.find_first_of(delim,s);
 
  if(s != e)
    {                   // non-empty field
      insert(end(),str.substr(s,e-s));
    }
  else
    {                   // empty field
      insert(end(),"");
    }

  if(e != string::npos)
    {                   // continue...
      parse(str.substr(e+1,string::npos),delim,spacer);
    }

  return;                // end of the string
}

const string StringList::dump(const char delim)
{
  if(size() == 0)
    return "";

  vector<string>::iterator i = begin();
  string str(*i);
  for(++i ; i != end() ; ++i)
    {
      str += delim;
      str += *i;
    }
  return str;
}




#ifdef __STRING_LIST_DEBUG__

main()
{
  StringList a("a b c d e");
  cerr << "expects: a!b!c!d!e" << endl;
  cerr << "result:  " << a.dump('!') << endl;

  StringList b("a: b: c :d: e:",":"," ");
  cerr << "expects: a!b!c !d!e" << endl;
  cerr << "result:  " << b.dump('!') << endl;

  StringList c(":a: b:: c: : d : e:",":"," ");
  cerr << "expects: !a!b!!c!!d !e!" << endl;
  cerr << "result:  " << c.dump('!') << endl;

}


#endif /*  __STRING_LIST_DEBUG__ */
