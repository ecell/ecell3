//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 2000-2001 Keio University
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
// written by Masayuki Okayama <smash@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
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

  class DataPoint;
  class DataPointStlVector;


  // Destructor

  DataPointStlVector::~DataPointStlVector( void )
  {
    for( iterator i( begin() ) ; i < end(); i++ )
      {
	delete *i;
      }
  }

  //

  void DataPointStlVector::push( RealCref t,
				 UVariableCref val)
  {
    theContainer->push_back( new Containee( t, val ) );
  }

  void DataPointStlVector::push(DataPointStlVector::ContaineeCref x)
  {
    theContainer->push_back( new Containee( x ) );
  }

  //


  DataPointStlVector::const_iterator 
  DataPointStlVector::
  binary_search(const_iterator first,
		const_iterator last,
		RealCref aTime) const
  {
    DataPoint dp( aTime, aTime );
    const_iterator itr = lower_bound( first, last, &dp );
    return itr;
  }


} // namespace libecs


#if defined(STLDATAPOINTVECTOR_TEST)

#include <iostream>
#include "DataPoint.cpp"

using namespace libecs;

typedef double Real;

int main()
{
  DataPoint<Real,Real> dp1 = DataPoint<Real,Real>(3.14,3.14);
  DataPointStlVector<Real,Real> dpvec = DataPointStlVector<Real,Real>(); 
  dpvec.push(0,0);
  dpvec.push(dp1);
  dpvec.push(3.15,3.0);
  dpvec.push(8.5,3.1);
  dpvec.push(100.45, 1.0);
  DataPointStlVector<Real,Real> dpvec_clone = DataPointStlVector<Real,Real>(dpvec); 
  
  DataPointStlVector<Real,Real>::iterator i;
  for(i=dpvec_clone.begin();i<dpvec_clone.end();i++)
    {
      printf("%p getTime = %f, getValue = %f\n",i,(*i)->getTime(),(*i)->getValue());
    }
  for(i=dpvec.begin();i<dpvec.end();i++)
    {
      printf("%p getTime = %f, getValue = %f\n",i,(*i)->getTime(),(*i)->getValue());
    }
  dpvec.binary_search(dpvec.begin(),dpvec.end(),0.4);

}

#endif /* END OF STLDATAPOINTVECTOR_TEST */






