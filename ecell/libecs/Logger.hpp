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


/*

 */

#include "DataPoint.hpp"
#include "MessageInterface.hpp"
#include "UniversalVariable.hpp"

//#if defined(STLDATAPOINTVECTOR)
#include "StlDataPointVector.hpp"
//#endif /* END OF STLDATAPOINTVECTOR */

namespace libecs
{

  class DataPoint;
  class Logger;



  /**
   
   */

  class Logger
  {




  public:

    //#if defined(STLDATAPOINTVECTOR)
    DECLARE_TYPE( StlDataPointVector, DataPointVector );
    //#endif /* END OF STLDATAPOINTVECTOR */

#if defined(VVECTOR)
    DECLARE_TYPE( VVector, DataPointVector );
#endif /* END OF VVECTOR */ 

    typedef DataPointVector::Containee containee_type;
    typedef DataPointVector::const_iterator const_iterator;
    typedef DataPointVector::iterator iterator;
    typedef DataPointVector::size_type size_type;

  typedef AbstractMessageSlot::ProxyMessageSlot ProxyMessageSlot;

  
  public:

    /**
       Default constructor
    */
  
    Logger( void );

    /**
       Constructor
    */
  
    Logger( const ProxyMessageSlot& );
  
    /**
       Copy constructor
    */
  
    Logger( LoggerCref );


    /// Destructor

    ~Logger( void );


    /**

     */

    DataPointVectorCref getData( void ) const;

    /**

     */

    DataPointVectorCref getData( RealCref start,
				 RealCref end ) const;

    /**

     */

    DataPointVectorCref getData( RealCref first,
				 RealCref last, 
				 RealCref interval ) const;


    /**

     */

    /*    void update( void ); */
  
    /**

     */
    //FIXME temp

    /*
    void update( containee_type& dp )
    {
      appendData(dp);
    }
    */


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



  protected:

    /**

     */

    const ProxyMessageSlot& getMessageSlot( void ) const
    {
      return theMessageSlot;
    }
  
  
    /**

     */

    DataPointVectorCref getDataPointVector( void ) const
    {
      return theDataPointVector;
    }


    /**

     */

    const_iterator binary_search( const_iterator begin,
				  const_iterator end,
				  RealCref t ) const
    {
      return theDataPointVector.binary_search( begin, end, t );
    }
    


    /**

     */

    void appendData( const containee_type& );

    void appendData( RealCref t, UniversalVariableCref v );

  private:


    /// Assignment operator is hidden
  
    Logger& operator=( const Logger& );
  



  private:
  

    //
    // Protected and Private data members follow
    //


  private:

    /// Data members

    DataPointVector             theDataPointVector;
    const ProxyMessageSlot      theMessageSlot;
    Real                        theMinimumInterval;
    Real                        theCurrentInterval;

  };

} // namespace libecs


#endif /* __LOGGER_HPP */

