//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
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
//



#include "DynamicPriorityQueue.hpp"



#ifdef DPQ_TEST

#include <iostream>

int main()
{
  DynamicPriorityQueue<int> dpq;


  for( int i( 10000 ); i != 0  ; --i )
    //  for( int i( 0 ); i < 10000  ; ++i )
    {
      dpq.push( i );
    }

  for( int i( 0 ); i < 10000  ; ++i )
    {
      dpq.changeOneKey( dpq.topIndex(), i+1000 );
    }

  DynamicPriorityQueue<int> copy( dpq );
  while( ! copy.empty() )
    {

      std::cerr << copy.top() << std::endl;
      copy.pop();

    }


}


#endif /* DPQ_TEST */
