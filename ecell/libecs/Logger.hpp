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


#if !defined(__LOGGER_HPP)
#define __LOGGER_HPP

#include "libecs.hpp"

#include "RootSystem.hpp"


/*

 */

#include "DataPoint.hpp"
#include "UVariable.hpp"

#include "DataPointStlVector.hpp"

namespace libecs
{

  /**
   
   */

  class Logger
  {
  public:

    DECLARE_TYPE( DataPointStlVector, DataPointVector );

    typedef Containee containee_type;
    typedef DataPointVector::const_iterator const_iterator;
    typedef DataPointVector::iterator iterator;


  public:

    /**
       Constructor

    */
  
    explicit Logger( const GetCurrentTimeMethodType& aGetCurrentTime,
		     PropertySlotRef aPropertySlot );

  
    /// Destructor

    ~Logger( void )
    {
      // self purge
      thePropertySlot.clearLogger();
    }


    /**

     */

    DataPointVectorCref getData( void ) const;

    /**

     */

    const DataPointVector getData( RealCref start,
				   RealCref end ) const;

    /**

     */

    const DataPointVector getData( RealCref first,
				   RealCref last, 
				   RealCref interval ) const;



    StringCref getName() const;

    /**

     */

    RealCref getStartTime( void ) const;

    /**

     */

    RealCref getEndTime( void ) const;


    /**

     */

    RealCref getMinInterval( void ) const
    {
      return theMinimumInterval;
    }

    /**

     */

    RealCref getCurrentInterval( void ) const
    {
      return theCurrentInterval;
    }



    /**

     */

    void appendData( RealCref v );


    /**

     */

    DataPointVectorCref getDataPointVector( void ) const
    {
      return theDataPointVector;
    }


  protected:

    /**


    PropertySlotCref getPropertySlot( void ) const
    {
      return thePropertySlotProxy;
    }
    */
  
  
    /**

     */

    const_iterator lower_bound( const_iterator begin,
				const_iterator end,
				RealCref t ) const
    {
      return theDataPointVector.lower_bound( begin, end, t );
    }
    
    const_iterator upper_bound( const_iterator begin,
				const_iterator end,
				RealCref t ) const
    {
      return theDataPointVector.upper_bound( begin, end, t );
    }
    


  private:

    // no copy constructor
  
    Logger( LoggerCref );

    /// Assignment operator is hidden
  
    Logger& operator=( const Logger& );

    /// no default constructor

    Logger( void );
  



  private:

    /// Data members

    PropertySlotRef      thePropertySlot;
    const GetCurrentTimeMethodType& theGetCurrentTimeMethod; 
    Real                 theMinimumInterval;
    Real                 theCurrentInterval;
    DataPointVector      theDataPointVector;

  };

} // namespace libecs


#endif /* __LOGGER_HPP */

