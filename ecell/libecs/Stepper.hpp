//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
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

#ifndef ___STEPPER_H___
#define ___STEPPER_H___

#include <vector>
#include <map>
#include <algorithm>
#include <utility>

#include "libecs.hpp"

#include "Util.hpp"
#include "UVariable.hpp"
#include "System.hpp"



namespace libecs
{

  /* *defgroup libecs_module The Libecs Module 
   * This is the libecs module 
   * @{ 
   */ 
  
  DECLARE_CLASS( Euler1Stepper );
  DECLARE_CLASS( RungeKutta4Stepper );
  DECLARE_CLASS( PropertySlotVectorImplementation );

  DECLARE_VECTOR( SubstancePtr, SubstanceVector );
  DECLARE_VECTOR( ReactorPtr,   ReactorVector );
  DECLARE_VECTOR( SystemPtr,    SystemVector );
  
  typedef IntegratorPtr ( *IntegratorAllocator_ )( SubstanceRef );
  DECLARE_TYPE( IntegratorAllocator_, IntegratorAllocator );

  typedef StepperPtr (* StepperAllocator_ )();
  DECLARE_TYPE( StepperAllocator_, StepperAllocator );

  DECLARE_VECTOR( StepperPtr, StepperVector );

  DECLARE_VECTOR( PropertySlotPtr, PropertySlotVector );



  /**
     A Stepper class defines and governs computation unit in a model.



  */

  class Stepper
  {

  public:

    Stepper(); 
    virtual ~Stepper() 
    {
      ; // do nothing
    }

    /**

    */

    virtual void initialize();

    /**


    */

    void sync();

    /**

    */

    void step() 
    { 
      compute();
      theCurrentTime += getStepInterval();
    }

    /**

    */

    void push();

    
    /**

     @note each stepper class defines this

    */

    virtual void compute() = 0;



    /**

    @param aParameterList
    */

    virtual void setParameterList( UVariableVectorCref aParameterList );


    /**


       @param aSystem
    */

    void registerSystem( SystemPtr aSystem );

    /**


       @param aSystem
    */

    void removeSystem( SystemPtr aSystem );


    /**
       Get the current time of this Stepper.

       The current time is defined as a next scheduled point in time
       of this Stepper.

       @return the current time in Real.
    */

    RealCref getCurrentTime() const
    {
      return theCurrentTime;
    }

    /**

    This may be overridden in dynamically scheduled steppers.

    */

    void setStepInterval( RealCref aStepInterval );


    /**
       Get the step interval of this Stepper.

       The step interval is a length of time that this Stepper proceeded
       in the last step.
       
       @return the step interval of this Stepper
    */

    RealCref getStepInterval() const
    {
      return theStepInterval;
    }

    /**
       Get the number of steps per a second.

       'StepsPerSecond' is defined as 1.0 / getStepInterval().

       If you need to get a reciprocal of the step interval,
       use of this is more efficient than just getStepInterval(), because
       it is pre-calculated when the setStepInterval() is called.


       @return the number of steps per a second. (== 1.0 / getStepInterval )
    */

    RealCref getStepsPerSecond() const
    {
      return theStepsPerSecond;
    }

    //@{

    bool isEntityListChanged() const
    {
      return theEntityListChanged;
    }

    void setEntityListChanged()
    {
      theEntityListChanged = true;
    }

    void clearEntityListChanged()
    {
      theEntityListChanged = false;
    }

    //@}


    void registerSlaves( SystemPtr );

    void registerPropertySlotWithProxy( PropertySlotPtr );

    void registerLoggedPropertySlot( PropertySlotPtr );


    StringCref getName() const
    {
      return theName;
    }

    void setName( StringCref aName )
    {
      theName = aName;
    }

    void setMinInterval( RealCref aValue )
    {
      theMinInterval = aValue;
    }

    RealCref getMinInterval() const
    {
      return theMinInterval;
    }

    void setMaxInterval( RealCref aValue )
    {
      theMaxInterval = aValue;
    }

    RealCref getMaxInterval() const
    {
      return theMaxInterval;
    }


    bool operator<( StepperCref rhs )
    {
      return getCurrentTime() < rhs.getCurrentTime();
    }

    virtual StringLiteral getClassName() const  { return "Stepper"; }


  protected:


    SystemVectorCref getSystemVector() const
    {
      return theSystemVector;
    }


    void setCurrentTime( RealCref aTime )
    {
      theCurrentTime = aTime;
    }

    void calculateStepsPerSecond();


    void searchSlaves( SystemPtr aStartSystemPtr );


  protected:

    SystemVector        theSystemVector;

    PropertySlotVector  thePropertySlotWithProxyVector;
    PropertySlotVector  theLoggedPropertySlotVector;

  private:

    Real                theCurrentTime;

    Real                theStepInterval;
    Real                theStepsPerSecond;

    Real                theMinInterval;
    Real                theMaxInterval;

    String              theName;

    bool                theEntityListChanged;

  };


  template< class Base_, class Derived_ = Base_ >
  class EntityCache
    :
    public std::vector<Derived_*>
  {

  public:

    DECLARE_TYPE( Base_, Base );
    DECLARE_TYPE( Derived_, Derived );


    //    typedef std::map<const String, BasePtr> BaseMap_;
    typedef std::vector<DerivedPtr> DerivedVector_;
    //    DECLARE_TYPE( BaseMap_, BaseMap );
    DECLARE_TYPE( DerivedVector_, DerivedVector );


    //    typedef std::less<const String> StringLess;
    //    DECLARE_MAP( const String, BasePtr, StringLess, BaseMap );

    EntityCache()
    {
      ; // do nothing
    }

    ~EntityCache() {}


    /**
       Update the cache.

    */

    void update( SystemVectorCref aSystemVector )
    {
      clear();

      for( SystemVectorConstIterator i( aSystemVector.begin() );
	   i != aSystemVector.end() ; ++i )
	{
	  const SystemCptr aSystem( *i );

	  typedef std::map<const String,BasePtr> BaseMap;
	  
	  for( typename BaseMap::const_iterator 
		 j( aSystem->System::getMap<Base>().begin() );
	       j != aSystem->System::getMap<Base>().end(); ++j )
	    {
	      BasePtr aBasePtr( (*j).second );

	      try
		{
		  DerivedPtr 
		    aDerivedPtr( dynamic_cast<DerivedPtr>( aBasePtr ) );
		  push_back( aDerivedPtr );
		}
	      catch( const std::bad_cast& )
		{
		  ; // do nothing
		}
	    }

	}

    }

  };


  DECLARE_CLASS( SRMSubstance );

  class SRMStepper 
    : 
    public Stepper
  {
  public:

    typedef EntityCache<Substance,SRMSubstance> SRMSubstanceCache;
    typedef EntityCache<Reactor> ReactorCache;

    SRMStepper();
    virtual ~SRMStepper() {}

    virtual void compute()
    {
      clear();
      differentiate();
      integrate();
      //???();

    }

    virtual void initialize();

    virtual void clear();
    virtual void differentiate();
    virtual void turn();
    virtual void integrate();
    //    virtual void ???();

    virtual StringLiteral getClassName() const  { return "SRMStepper"; }
 

  protected:

    IntegratorAllocator theIntegratorAllocator;

    SRMSubstanceCache theSubstanceCache;
    ReactorCache      theReactorCache;

  private:

    virtual void distributeIntegrator( IntegratorAllocator );

  };

  class Euler1SRMStepper
    :
    public SRMStepper
  {

  public:

    Euler1SRMStepper();
    virtual ~Euler1SRMStepper() {}

    static StepperPtr createInstance() { return new Euler1SRMStepper; }

    virtual StringLiteral getClassName() const  { return "Euler1SRMStepper"; }
 
  protected:

    static IntegratorPtr newIntegrator( SRMSubstanceRef substance );

  };


  class RungeKutta4SRMStepper
    : 
    public SRMStepper
  {

  public:

    RungeKutta4SRMStepper();
    virtual ~RungeKutta4SRMStepper() {}

    static StepperPtr createInstance() { return new RungeKutta4SRMStepper; }

    virtual void differentiate();

    virtual StringLiteral getClassName() const 
    { return "RungeKutta4SRMStepper"; }

  private:

    static IntegratorPtr newIntegrator( SRMSubstanceRef substance );

  };



} // namespace libecs

#endif /* ___STEPPER_H___ */



/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
