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

#ifndef ___SYSTEM_H___
#define ___SYSTEM_H___
#include <map>

#include "libecs.hpp"

#include "Entity.hpp"
#include "Reactor.hpp"

namespace libecs
{

  // Tree data structures used for entry lists
  // for_each performance is very important. other or hybrid container type?
  DECLARE_MAP( const String, SubstancePtr, less<const String>, SubstanceMap );
  DECLARE_MAP( const String, ReactorPtr,   less<const String>, ReactorMap );
  DECLARE_MAP( const String, SystemPtr,    less<const String>, SystemMap );

  typedef SystemPtr (*SystemAllocatorFunc)();

  class System : public Entity
  {

  public: 

    class isRegularReactorItem;

  public:

    System();
    virtual ~System();

    virtual const PrimitiveType getPrimitiveType() const
    {
      return PrimitiveType( PrimitiveType::SYSTEM );
    }

    virtual void initialize();

    virtual void clear();
    virtual void react();
    virtual void turn(); 
    virtual void transit();
    virtual void postern();

    virtual const char* const className() const { return "System"; }

    /**
       Set a pointer to the RootSystem.
       Usually no need to use this because
       setSuperSystem() will do this automatically.

       @return the pointer to the RootSystem.
    */
    void setRootSystem( RootSystemPtr rootsystem ) 
    { 
      theRootSystem = rootsystem; 
    }

    /**
       Get a pointer to the RootSystem that this System belongs.
       Unlike other Primitive classes, System objects must the pointer
       to the RootSystem.

       @return the pointer to the RootSystem.
    */
    RootSystemPtr getRootSystem() const { return theRootSystem; }

    /**
       Set supersystem of this System.
       Unlike other Primitive classes, theRootSystem is also set 
       in this method as well as theSupersystem.

       @param supersystem a pointer to a System to which this object belongs.
    */
    void setSuperSystem( SystemPtr const supersystem );

    /**
       @return A pointer to a Stepper object that this System has or
       NULL pointer if it is not set.
    */
    StepperPtr getStepper() const { return theStepper; }

    /**
       Instantiate a Stepper object of @a classname using theRootSystem's
       StepperMaker object.  Register the Stepper object as a stepper for 
       this System.

       @param classname Classname of the Stepper that this System may have.
    */
    void setStepper( StringCref classname );

    /**
       @return Volume of this System. Unit is [L].
    */
    virtual RealCref getVolume()
    {
      return theVolume;
    }

    /**
       Set a new volume for this System. 
       Make the new volume effective from beginning of next time step.
     */
    void setVolume( const Real volume )
    {
      theVolumeBuffer = volume;
    }

    /**
       Register a Reactor object in this System.
    */
    void registerReactor( ReactorPtr const newone );
  
    /**
       @return true: if this System contains a Reactor whose name is @a id.
    */
    bool containsReactor( StringCref id )
    {
      return ( getReactorIterator( id ) != theReactorMap.end() ) 
	? true : false;
    }

    /**
       @return An iterator which points to the first Reactor in this System.
    */
    ReactorMapIterator getFirstReactorIterator()
    {
      return theReactorMap.begin();
    }

    /**
       @return An iterator which points to the first regular Reactor
       (i.e. not posterior Reactor) in this System.
    */
    ReactorMapIterator getFirstRegularReactorIterator() const
    {
      return theFirstRegularReactorIterator;
    }

    /**
       @return An iterator which points to the last Reactor in this System.
    */
    ReactorMapIterator getLastReactorIterator() 
    { 
      return theReactorMap.end();
    }

    /**
       @return An iterator which points to a Reactor whose name is @a id.
    */
    ReactorMapIterator getReactorIterator( StringCref id )
    {
      return theReactorMap.find( id );
    }

    /**
       @return The number of Reactors in this object.
    */
    int getNumberOfReactors() const
    {
      return theReactorMap.size();
    }

    /**
       Find a Reactor with given id. Unlike getReactorIterator(), this 
       throws NotFound exception if it is not found.

       @return An pointer to a Reactor object in this System named @a id.
    */
    ReactorPtr getReactor( StringCref id ) ;

    /**
       Register a Substance object in this System.
    */
    void registerSubstance( SubstancePtr id );
  
    /**
       @return An iterator which points to the first Substance in this System.
    */
    SubstanceMapIterator getFirstSubstanceIterator()
    {
      return theSubstanceMap.begin();
    }

    /**
       @return An iterator which points to the last Substance in this System.
    */
    SubstanceMapIterator getLastSubstanceIterator()
    {
      return theSubstanceMap.end();
    }

    /**
       @return An iterator which points to a Substance whose name is @a id.
    */
    SubstanceMapIterator getSubstanceIterator( StringCref id )
    {
      return theSubstanceMap.find( id );
    }

    /**
       @return true: if this System contains a Substance whose name is @a id.
    */
    bool containsSubstance( StringCref id )
    {
      return ( getSubstanceIterator( id ) != theSubstanceMap.end() ) ?
	true : false;
    }

    /**
       @return The number of Substances in this object.
    */
    int getNumberOfSubstances() const
    {
      return theSubstanceMap.size();
    }

    /**
       @return An pointer to a Substance object in this System named @a id.
    */
    SubstancePtr getSubstance( StringCref id );

    /**
       Register a System object in this System
    */
    void registerSystem( SystemPtr );

    /**
       @return An iterator which points to the first System in this System.
    */
    SystemMapIterator getFirstSystemIterator()
    {
      return theSystemMap.begin();
    }

    /**
       @return An iterator which points to the last System in this System.
    */
    SystemMapIterator getLastSystemIterator()
    {
      return theSystemMap.end();
    }

    /**
       @return An iterator which points to a System whose name is @a id.
    */
    SystemMapIterator getSystemIterator( StringCref id )
    {
      return theSystemMap.find( id );
    }

    /**
       @return true: if this System contains a System whose name is @a id.
    */
    bool containsSystem( StringCref id )
    {
      return ( getSystemIterator( id ) != theSystemMap.end() ) ? 
	true : false;
    }

    /**
       @return The number of Systems in this object.
    */
    int getNumberOfSystems() const
    {
      return theSystemMap.size();
    }

    /**
       @return An pointer to a System object in this System whose ID is id.
    */
    virtual SystemPtr getSystem( StringCref id );

    /**
       This method finds recursively a System object pointed by
       @a SystemPath

       @return An pointer to a System object in this or subsystems of this
       System object pointed by @a SystemPath
    */
    virtual SystemPtr getSystem( SystemPathCref systempath );

    virtual EntityPtr getEntity( FullIDCref fullid );

    virtual void createEntity( StringCref classname,
			       FullIDCref fullid,
			       StringCref name );


    Real getActivityPerSecond();

    RealCref getStepInterval() const;
    RealCref getStepsPerSecond() const;


    static SystemPtr instance() { return new System; }

  public: // message slots

    void setStepper( MessageCref message );
    void setVolume( MessageCref message);

    const Message getStepper( StringCref keyword );

    const Message getSystemList( StringCref keyword );
    const Message getSubstanceList( StringCref keyword );
    const Message getReactorList( StringCref keyword );

    const Message getVolume( StringCref keyword );
    const Message getStepInterval( StringCref keyword );

  protected:

    virtual void makeSlots();

    void updateVolume()
    {
      theVolume = theVolumeBuffer;
    }

  protected:

    StepperPtr theStepper;

    ReactorMapIterator theFirstRegularReactorIterator;

  private:

    Real theVolume;
    Real theVolumeBuffer;

    ReactorMap   theReactorMap;
    SubstanceMap theSubstanceMap;
    SystemMap    theSystemMap;

    RootSystemPtr theRootSystem;

  };

  /**
  Equivalent to Reactor::isRegularReactor except that
  this function object takes a reference to a ReactorMap::value_type.
  */
  class System::isRegularReactorItem
    : public unary_function< const ReactorMap::value_type,bool >
  {
  public:
    bool operator()( const ReactorMap::value_type r ) const
    {
      return Reactor::isRegularReactor::isRegularName( ( r.second )->getID() );
    }
  };



} // namespace libecs

#endif /* ___SYSTEM_H___ */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
