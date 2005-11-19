//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2000-2002 Keio University
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
// written by Masayuki Okayama <smash@e-cell.org>,
// E-Cell Project.
//


/*





 */

#if !defined(__DATAPOINTVECTOR_STL_HPP)
#include "DataPointStlVector.hpp"
#endif

#include <algorithm>
#include <stdio.h> // FIXME : for debugging



namespace libecs
{

  // Destructor

  DataPointStlVector::~DataPointStlVector( void )
  {
    /*
    for( iterator i( theContainer.begin() ) ; i < theContainer.end(); i++ )
      {
	delete *i;
      }
    */
  }

  //


} // namespace libecs


#if defined(DATAPOINTSTLVECTOR_TEST)

#include <iostream>
#include "DataPoint.cpp"
#include "libecs.hpp"

using namespace libecs;


int main()
{
  DataPoint dp1( DataPoint(3.14,3.14) );
  DataPointStlVector* dpvec(new DataPointStlVector()); 
  dpvec->push(0,0);
  dpvec->push(dp1);
  dpvec->push(3.15,3.0);
  dpvec->push(8.5,3.1);
  dpvec->push(100.45, 1.0);
  //  DataPointStlVector* dpvec_clone( new DataPointStlVector( *dpvec ) ); 
  DataPointStlVector dpvec_clone( *dpvec ); 
  


  for( DataPointStlVector::const_iterator i=dpvec_clone.begin();i<dpvec_clone.end();++i)
    {
      printf("%p getTime = %f, getValue = %f\n",i,(*i)->getTime(),(*i)->getValue());
    }
  for( DataPointStlVector::const_iterator i=dpvec->begin();i<dpvec->end();++i)
    {
      printf("%p getTime = %f, getValue = %f\n",i,(*i)->getTime(),(*i)->getValue());
    }

  dpvec->binary_search(dpvec->begin(),dpvec->end(),0.4);

}

#endif /* END OF DATAPOINTSTLVECTOR_TEST */






