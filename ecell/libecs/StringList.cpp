//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2000 Keio University
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
//
// written by Kouichi Takahashi <shafi@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//

#include "StringList.hpp"


namespace libecs
{

  StringList::StringList( StringCref str, StringCref delim,
			  StringCref spacer )
  {
    parse( str, delim, spacer );
  }

  void StringList::parse( StringCref str, StringCref delim,
			  StringCref spacer)
  {
    if( str == "" )
      {
	clear();
	return;
      }

    String::size_type   s = str.find_first_not_of( spacer );
    String::size_type   e = str.find_first_of( delim, s );
 
    // non-empty field
    if( s != e )
      {                   
	insert( end(), str.substr( s, e - s ) );
      }
    // empty field
    else
      {                   
	insert( end(), "" );
      }

    // continue...
    if( e != String::npos )
      { 
	parse( str.substr( e + 1, String::npos ), delim, spacer );
      }

    // end of the string
    return; 
  }

  const String StringList::dump( const char delim )
  {
    if( size() == 0 )
      {
	return "";
      }

    //vector< String >::iterator i = begin();
    iterator i = begin();
    String str( *i );
    for( ++i ; i != end() ; ++i )
      {
	str += delim;
	str += *i;
      }
    return str;
  }


} // namespace libecs



#ifdef __STRINGLIST_TEST__

using namespace libecs;

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
