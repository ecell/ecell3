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
  DECLARE_MAP( const String, SubstancePtr, less<const String>, SubstanceList );
  DECLARE_MAP( const String, ReactorPtr,   less<const String>, ReactorList );
  DECLARE_MAP( const String, SystemPtr,    less<const String>, SystemList );

  typedef SystemPtr (*SystemAllocatorFunc)();

  class System : public Entity
  {

  public: 

    class isRegularReactorItem;

  public:

    System();
    virtual ~System();

    const String getFqpi() const;

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
    StepperPtr getStepper() { return theStepper; }

    /**
       Instantiate a Stepper object of @a classname using theRootSystem's
       StepperMaker object.  Register the Stepper object as a stepper for 
       this System.

       @param classname Classname of the Stepper that this System may have.
    */
    void setStepper( StringCref classname );

    /**
       This method takes a FQID of a Reactor as a VolumeIndex of this System.
       The FQID will be resolved to ReactorPtr at initialize().

       @param fqen FQID of a VolumeIndex Reactor for this System.
    */
    void setVolumeIndex( const FQID& fqen );

    /**
       @return a pointer to the VolumeIndex Reactor of this System.
    */
    ReactorPtr getVolumeIndex() { return theVolumeIndex; }

    /**
       Volume of a System is calculated by activity() of
       VolumeIndex Reactor of the System.

       @return Volume of this System. Unit is [L].
    */
    virtual Real getVolume();

    /**
       Add a Reactor object in this RSystem.
    */
    void addReactor( ReactorPtr const newone );
  
    /**
       @return true: if this System contains a Reactor whose name is @a id.
    */
    bool containsReactor( StringCref id )
    {
      return ( getReactorIterator( id ) != theReactorList.end() ) 
	? true : false;
    }

    /**
       @return An iterator which points to the first Reactor in this System.
    */
    ReactorListIterator getFirstReactorIterator()
    {
      return theReactorList.begin();
    }

    /**
       @return An iterator which points to the first regular Reactor
       (i.e. not posterior Reactor) in this System.
    */
    ReactorListIterator getFirstRegularReactorIterator() const
    {
      return theFirstRegularReactorIterator;
    }

    /**
       @return An iterator which points to the last Reactor in this System.
    */
    ReactorListIterator getLastReactorIterator() 
    { 
      return theReactorList.end();
    }

    /**
       @return An iterator which points to a Reactor whose name is @a id.
    */
    ReactorListIterator getReactorIterator( StringCref id )
    {
      return theReactorList.find( id );
    }

    /**
       @return The number of Reactors in this object.
    */
    int getNumberOfReactors() const
    {
      return theReactorList.size();
    }

    /**
       Find a Reactor with given id. Unlike getReactorIterator(), this 
       throws System::NotFound exception if it is not found.

       @return An pointer to a Reactor object in this System named @a id.
    */
    ReactorPtr getReactor( StringCref id ) ;

    /**
       Add a Substance object in this System.
    */
    void addSubstance( SubstancePtr id );
  
    /**
       @return An iterator which points to the first Substance in this System.
    */
    SubstanceListIterator getFirstSubstanceIterator()
    {
      return theSubstanceList.begin();
    }

    /**
       @return An iterator which points to the last Substance in this System.
    */
    SubstanceListIterator getLastSubstanceIterator()
    {
      return theSubstanceList.end();
    }

    /**
       @return An iterator which points to a Substance whose name is @a id.
    */
    SubstanceListIterator getSubstanceIterator( StringCref id )
    {
      return theSubstanceList.find( id );
    }

    /**
       @return true: if this System contains a Substance whose name is @a id.
    */
    bool containsSubstance( StringCref id )
    {
      return ( getSubstanceIterator( id ) != theSubstanceList.end() ) ?
	true : false;
    }

    /**
       @return The number of Substances in this object.
    */
    int getNumberOfSubstances() const
    {
      return theSubstanceList.size();
    }

    /**
       @return An pointer to a Substance object in this System named @a id.
    */
    SubstancePtr getSubstance( StringCref id );

    /**
       Add a System object in this System
    */
    void addSystem( SystemPtr );

    /**
       @return An iterator which points to the first System in this System.
    */
    SystemListIterator getFirstSystemIterator()
    {
      return theSubsystemList.begin();
    }

    /**
       @return An iterator which points to the last System in this System.
    */
    SystemListIterator getLastSystemIterator()
    {
      return theSubsystemList.end();
    }

    /**
       @return An iterator which points to a System whose name is @a id.
    */
    SystemListIterator getSystemIterator( StringCref id )
    {
      return theSubsystemList.find( id );
    }

    /**
       @return true: if this System contains a System whose name is @a id.
    */
    bool containsSystem( StringCref id )
    {
      return ( getSystemIterator( id ) != theSubsystemList.end() ) ? 
	true : false;
    }

    /**
       @return The number of Systems in this object.
    */
    int getNumberOfSystems() const
    {
      return theSubsystemList.size();
    }

    /**
       @return An pointer to a System object in this System whose ID is id.
    */
    virtual SystemPtr getSystem( StringCref id ) ;

    /**
       This method finds recursively a System object pointed by
       @a systempath.

       @return An pointer to a System object in this or subsystems of this
       System object pointed by @a systempath
    */
    virtual SystemPtr getSystem( SystemPathCref systempath ) ;


  public: // message interfaces

    void setStepperClass( MessageCref message );
    void setVolumeIndex( MessageCref message );

    const Message getStepperClass( StringCref keyword );
    const Message getVolumeIndex( StringCref keyword );

    const Message getSystemList( StringCref keyword );
    const Message getSubstanceList( StringCref keyword );
    const Message getReactorList( StringCref keyword );

  protected:

    virtual void makeSlots();

  protected:

    FQIDPtr    theVolumeIndexName;
    ReactorPtr theVolumeIndex;
    StepperPtr theStepper;

    ReactorListIterator theFirstRegularReactorIterator;

  private:

    ReactorList   theReactorList;
    SubstanceList theSubstanceList;
    SystemList    theSubsystemList;

    RootSystemPtr theRootSystem;

  };

  /**
  Equivalent to Reactor::isRegularReactor except that
  this function object takes a reference to a ReactorList::value_type.
  */
  class System::isRegularReactorItem
    : public unary_function< const ReactorList::value_type,bool >
  {
  public:
    bool operator()( const ReactorList::value_type r ) const
    {
      return Reactor::isRegularReactor::isRegularName( ( r.second )->getId() );
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
