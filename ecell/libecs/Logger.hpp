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
// modified by Gabor Bereczki <gabor.bereczki@talk21.com>



#if !defined(__LOGGER_HPP)
#define __LOGGER_HPP

#include "libecs.hpp"
#include "Model.hpp"
#include "UVariable.hpp"
#include "PhysicalLogger.hpp"
#include "DataPointVector.hpp"

namespace libecs
{


  /** \defgroup logging The Data Logging Module.
   * The Data Logging Module.
   * @{ 
   */ 


  /**
   
  */

  class Logger
  {
  public:

    typedef DataPointVectorIterator const_iterator;


  public:

    /**
       Constructor

    */
  
    //    explicit Logger( ModelCref aModel, PropertySlotRef aPropertySlot );
    explicit Logger( PropertySlotRef aPropertySlot,
		     StepperCref aStepper, 
		     RealCref aMinimumInterval = 0.1 );
  
    /// Destructor

    ~Logger( void )
    {
      // self purge
    }


    /**

    */

    DataPointVectorRCPtr getData( void ) ;

    /**


    \note It is assumed for both following getData methods that the
       Time values returned by GetCurrentTime method are monotonously
       increasing, therefore
       - a newer theTime is always greater than previous
       - no 2 theTime values are the same 
    */

    DataPointVectorRCPtr getData( RealCref aStartTine,
				  RealCref anEndTime ) ;

    /**
    
    */

    DataPointVectorRCPtr getData( RealCref aStartTime,
				  RealCref anEndTime, 
				  RealCref anInterval ) ;
    


    /**

    */

    Real getStartTime( void ) ;

    /**

    */

    Real getEndTime( void ) ;


    const Int getSize() const
    {
      return thePhysicalLogger.size();
    }

    /**

    */

    RealCref getMinimumInterval( void ) const
    {
      return theMinimumInterval;
    }


    /**

    */

    void appendData( RealCref v );


    /**

    */

    void flush();


  protected:

    /**

    \internal

    */

    const_iterator binary_search( const_iterator begin,
				  const_iterator end,
				  RealCref t ) 
    {
      return thePhysicalLogger.lower_bound( thePhysicalLogger.begin(), 
    					    thePhysicalLogger.end(), 
					    t );
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

    StepperCref          theStepper;
    PropertySlotRef      thePropertySlot;

    DataPointVectorRCPtr theDataPointVector;
    PhysicalLogger	 thePhysicalLogger;
    DataInterval	 theDataInterval;
    Real                 theLastTime;

    Real                 theMinimumInterval;

  };


  /** @} */ // logging module

} // namespace libecs


#endif /* __LOGGER_HPP */

