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


#if !defined(__DATAPOINT_HPP)
#define __DATAPOINT_HPP

#include "libecs.hpp"


#include "UVariable.hpp"

namespace libecs
{


/*

 */

/**

 */


  class DataPoint
  {
    
    
  public:
    
    
    /**
       Initializing constructor
       @param 2 objects which are components of DataPoint
    */
    
    explicit DataPoint( RealCref, UVariableCref );
    
    explicit DataPoint( UVariableCref, UVariableCref );
    
    explicit DataPoint( UVariableCref, RealCref );

    explicit DataPoint( RealCref, RealCref );

    
    /**
       Copy constructor
       @param Object constant reference
    */
    
    DataPoint( DataPointCref );
    
    
    /**
       Copy constructor
       @param Object constant reference
    */
    
    DataPoint( DataPointRef );
    
    
    /// Destructor
    
    ~DataPoint( void )
    {
      ; // do nothing
    }
    
    
    /**
       Assignments operator
       @param DataPoint constant reference
       @return DataPoint reference
    */
    


    // FIXME
    bool operator<( DataPointCref second )
    {
      if( getTime() < second.getTime() )
	{
	  return true;
	}
      return false;
    }
    
    // FIXME
    bool operator>(DataPointCref second)
    {
      if( getTime() > second.getTime() )
	{
	  return true;
	}
      return false;
    }
    
    
    //
    // Accessors
    //
    
    /**
       Return the data member, theTime
       @return T constant reference
    */
    
    RealCref getTime( void ) const
    {
      return theTime;
    }
    
    
    /**
       Return the data member, theValue
       @return V constant reference
    */
    
    RealCref getValue( void ) const
    {
      return theValue;
    }
    
  private:
    
    /// Default constructor prohibited to public use
    
    DataPoint( void );
    
    /**
       
       @param int object
       @return DataPoint reference
    */
    
    DataPoint& operator[]( int );
    
    //    DataPointRef operator=( DataPointCref );
    
    //
    // Mutators
    //
    
    //
    // Private data members follow
    //
    
  private:
    
    /// The internal value
    
    Real theTime;
    Real theValue;
    
  };
  
  

} // namespace libecs


#endif /* __DATAPOINT_HPP */
