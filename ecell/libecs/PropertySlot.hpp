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

#ifndef __PROPERTYSLOT_HPP
#define __PROPERTYSLOT_HPP

#include "libecs.hpp"
#include "Util.hpp"

#include "UVariable.hpp"


namespace libecs
{

  /** @defgroup libecs_module The Libecs Module 
   * This is the libecs module 
   * @{ 
   */ 
  
  template< typename FromType, typename ToType >
  ToType convertTo( const FromType& aValue )
  {
    convertTo( aValue, Type2Type< ToType >() );
  }

  template< typename FromType, typename ToType >
  ToType convertTo( const FromType& aValue, Type2Type<ToType> )
  {
    //    DefaultSpecializationInhibited();
#warning "unexpected default specialization of convertTo()"
    return ToType( aValue );
  }


  // to UVariableVectorRCPtr

  // identity

  template<>
  inline const UVariableVectorRCPtr 
  convertTo( UVariableVectorRCPtrCref aValue, 
	     Type2Type< const UVariableVectorRCPtr > )
  {
    return aValue;
  }

  // from Real
  template<>
  inline const UVariableVectorRCPtr 
  convertTo( RealCref aValue,
	     Type2Type< const UVariableVectorRCPtr > )
  {
    UVariableVectorRCPtr aVector( new UVariableVector );
    aVector->push_back( aValue );
    return aVector;
  }

  // from String
  template<>
  inline const UVariableVectorRCPtr 
  convertTo( StringCref aValue,
	     Type2Type< const UVariableVectorRCPtr > )
  {
    UVariableVectorRCPtr aVectorPtr( new UVariableVector );
    aVectorPtr->push_back( aValue );
    return aVectorPtr;
  }

  // to String

  // identity
  template<>
  inline const String
  convertTo( StringCref aValue,
	     Type2Type< const String > )
  {
    return aValue;
  }



  template<>
  inline const String convertTo( UVariableVectorRCPtrCref aValue,
				 Type2Type< const String > )
  {
    return (*aValue)[0].asString();
  }

  template<>
  inline const String convertTo( RealCref aValue,
				 Type2Type< const String > )
  {
    return toString( aValue );
  }


  // to Real


  // identity
  template<>
  inline const Real
  convertTo( RealCref aValue,
	     Type2Type< const Real > )
  {
    return aValue;
  }

  template<>
  inline const Real convertTo( UVariableVectorRCPtrCref aValue,
			       Type2Type< const Real > )
  {
    return (*aValue)[0].asReal();
  }
    
  template<>
  inline const Real convertTo( StringCref aValue,
			       Type2Type< const Real > )
  {
    return stringTo< Real >( aValue );
  }


#if 1
  DECLARE_CLASS(PropertySlotProxy);
  DECLARE_VECTOR(PropertySlotProxyPtr, PropertySlotProxyVector);
#endif //0

  /**
     Base class for PropertySlot classes.

     @see PropertyInterface
     @see Message
  */

  //  class PropertySlotBase
  class PropertySlot
  {

  public:

    //    PropertySlotBase( StringCref aName )
    PropertySlot( StringCref aName )
      :
      theName( aName ),
      //      // FIXME: dummyLogger ?
      theLogger( NULLPTR )
    {
      ; // do nothing
    }
    
    //    virtual ~PropertySlotBase()
    virtual ~PropertySlot()
    {
      ; // do nothing
    }

    virtual void setUVariableVectorRCPtr( UVariableVectorRCPtrCref ) = 0;
    virtual const UVariableVectorRCPtr getUVariableVectorRCPtr() const = 0;
    
    virtual void setReal( RealCref real ) = 0;
    virtual const Real getReal() const = 0;

    virtual void setString( StringCref string ) = 0;
    virtual const String getString() const = 0;

    virtual const bool isSetable() const = 0;
    virtual const bool isGetable() const = 0;

    virtual void sync() = 0;


    virtual void push()
    {
      if( isLogged() )
	{
	  updateLogger();
	}
      //      updateProxies();
    }


    //    virtual PropertySlotProxyPtr createProxy( void );

    

    StringCref getName() const
    {
      return theName;
    }

    const bool isLogged()
    {
      return theLogger != NULLPTR;
    }

    void connectLogger( LoggerPtr logger );

    void disconnectLogger();

    LoggerCptr getLogger() const
    {
      return theLogger;
    }

    void updateLogger();


  protected:

    String                   theName;
    LoggerPtr                theLogger;
    PropertySlotProxyVector  theProxyVector;

  };




  class FixPolicy
  {


  };


  class CanBeFixed
  {


  };

  class CannotBeFixed
  {


  };



  /**

  Determines when and how the property value is updated.
  
  Following three methods must be defined in subclass.
  
  GetType get() const
  void set( SetType aValue )
  void sync()
  
  */

  template 
  < 
    class T,
    typename SlotType_ 
    >
  class UpdatePolicy
  {

  public:


    DECLARE_TYPE( SlotType_, SlotType );

    typedef const SlotType GetType;
    typedef SlotTypeCref SetType;

    typedef GetType ( T::* GetMethodPtr )() const;
    typedef void ( T::* SetMethodPtr )( SlotTypeCref );

    typedef GetType ( UpdatePolicy::* CallGetMethodPtr )() const;
    typedef void ( UpdatePolicy::* CallSetMethodPtr )( SetType );

    UpdatePolicy( T& anObject,
		  const SetMethodPtr aSetMethodPtr,
		  const GetMethodPtr aGetMethodPtr )
      :
      theObject( anObject ),
      theSetMethodPtr( aSetMethodPtr ),
      theGetMethodPtr( aGetMethodPtr ),
      theCallSetMethodPtr( &UpdatePolicy::callNullSetMethod ),
      theCallGetMethodPtr( &UpdatePolicy::callNullGetMethod )
    {
      if( isSetable() )
	{
	  theCallSetMethodPtr = &UpdatePolicy::callSetMethodPtr;
	}

      if( isGetable() )
	{
	  theCallGetMethodPtr = &UpdatePolicy::callGetMethodPtr;
	}
    }

    ~UpdatePolicy()
    {
      ; // do nothing
    }

    const bool isSetable() const
    {
      return theSetMethodPtr != NULLPTR;
    }

    const bool isGetable() const
    {
      return theGetMethodPtr != NULLPTR;
    }


  protected:

    void callSetMethod( SetType aValue )
    {
      ( this->*theCallSetMethodPtr )( aValue );
    }

    GetType callGetMethod() const
    {
      return ( this->*theCallGetMethodPtr )();
    }


  private:

    void callSetMethodPtr( SetType aValue )    
    {
      ( theObject.*theSetMethodPtr )( aValue );
    }

    GetType callGetMethodPtr() const
    {
      return ( ( theObject.*theGetMethodPtr )() );
    }

    void callNullSetMethod( SetType )    
    {
      THROW_EXCEPTION( AttributeError, "Not setable." );
    }

    GetType callNullGetMethod() const
    {
      THROW_EXCEPTION( AttributeError, "Not getable." );
    }

  private:

    T& theObject;
    const SetMethodPtr theSetMethodPtr;
    const GetMethodPtr theGetMethodPtr;

    CallSetMethodPtr theCallSetMethodPtr;
    CallGetMethodPtr theCallGetMethodPtr;

  };



  template
  < 
    class T,
    typename SlotType_ 
    >
  class UpdateImmediately
    : 
    public UpdatePolicy< T, SlotType_ >
  {

  public:

    typedef typename UpdatePolicy< T, SlotType_ >::SetType SetType;
    typedef typename UpdatePolicy< T, SlotType_ >::GetType GetType;
    typedef typename UpdatePolicy< T, SlotType_ >::SetMethodPtr SetMethodPtr;
    typedef typename UpdatePolicy< T, SlotType_ >::GetMethodPtr GetMethodPtr;


    UpdateImmediately( T& anObject,
		       const SetMethodPtr aSetMethodPtr,
		       const GetMethodPtr aGetMethodPtr )
      : 
      UpdatePolicy< T, SlotType_ >( anObject, 
				    aSetMethodPtr,
				    aGetMethodPtr )
    {
      ; // do nothing
    }

    ~UpdateImmediately()
    {
      ; // do nothing
    }

    void set( SetType aValue )
    {
      callSetMethod( aValue );
    }

    GetType get() const
    {
      return callGetMethod();
    }

    void sync()
    {
      ; // do nothing
    }

  };

  template
  < 
    class T,
    typename SlotType_ 
    >
  class UpdateAtSync
    : 
    public UpdatePolicy< T, SlotType_ >
  {

  public:

    typedef typename UpdatePolicy< T, SlotType_ >::SetType SetType;
    typedef typename UpdatePolicy< T, SlotType_ >::GetType GetType;
    typedef typename UpdatePolicy< T, SlotType_ >::SetMethodPtr SetMethodPtr;
    typedef typename UpdatePolicy< T, SlotType_ >::GetMethodPtr GetMethodPtr;


    UpdateAtSync( T& anObject,
		  const SetMethodPtr aSetMethodPtr,
		  const GetMethodPtr aGetMethodPtr )
      : 
      UpdatePolicy< T, SlotType_ >( anObject, 
					aSetMethodPtr,
 					aGetMethodPtr )
    {
      ; // do nothing
    }

    ~UpdateAtSync()
    {
      ; // do nothing
    }

    void set( SetType aValue )
    {
      theValue = aValue;
    }

    GetType get() const
    {
      return callGetMethod();
    }

    void sync()
    {
      return callSetMethod( theValue );
    }

  private:

    SetType theValue;

  };


  template
  < 
    class T,
    typename SlotType_,
    template < class U, typename V > class UpdatePolicy_ = UpdateImmediately
  >
  class ConcretePropertySlot
    :
    //    public PropertySlotBase,
    public PropertySlot,
    public UpdatePolicy_< T, SlotType_ >
  {

  public:

    typedef UpdatePolicy_< T, SlotType_ > UpdatePolicy;

    DECLARE_TYPE( SlotType_, SlotType );
    typedef typename UpdatePolicy::SetType SetType;
    typedef typename UpdatePolicy::GetType GetType;
    typedef typename UpdatePolicy::SetMethodPtr SetMethodPtr;
    typedef typename UpdatePolicy::GetMethodPtr GetMethodPtr;


    ConcretePropertySlot( StringCref aName, T& anObject, 
			  const SetMethodPtr aSetMethodPtr,
			  const GetMethodPtr aGetMethodPtr )
      :
      //      PropertySlotBase( aName ),
      PropertySlot( aName ),
      UpdatePolicy( anObject, aSetMethodPtr, aGetMethodPtr )
    {
      ; // do nothing
    }

    virtual ~ConcretePropertySlot()
    {
      ; // do nothing
    }

    virtual void setUVariableVectorRCPtr( UVariableVectorRCPtrCref aValue )
    {
      setImpl( aValue );
    }

    virtual const UVariableVectorRCPtr getUVariableVectorRCPtr() const
    {
      return getImpl< const UVariableVectorRCPtr >();
    }

    virtual void setReal( RealCref aValue )
    {
      setImpl( aValue );
    }

    virtual const Real getReal() const
    {
      return getImpl< const Real >();
    }

    virtual void setString( StringCref aValue )
    {
      setImpl( aValue );
    }

    virtual const String getString() const
    {
      return getImpl< const String >();
    }

    virtual void sync()
    {
      UpdatePolicy::sync();
    }

    virtual const bool isSetable() const
    {
      return UpdatePolicy::isSetable();
    }

    virtual const bool isGetable() const
    {
      return UpdatePolicy::isGetable();
    }
      

  protected:

    template < typename TYPE >
    inline void setImpl( TYPE aValue )
    {
      UpdatePolicy::set( convertTo( aValue, Type2Type< SetType >() ) );
      //      UpdatePolicy::set( convertTo< TYPE, SetType >( aValue ) );
    }

    template < typename TYPE >
    inline TYPE getImpl() const
    {
      return convertTo( UpdatePolicy::get(), Type2Type< TYPE >() );
    }


  };




#if 0 
  class PropertySlotProxy
  {
    
  public:
      
    PropertySlotProxy( PropertySlotBasePtr aPropertySlot )
      :
      thePropertySlot( aPropertySlot )
    {
      ; // do nothing
    }

    // copy constructor
    
    PropertySlotProxy( PropertySlotProxyRef rhs )
      :
      thePropertySlot( rhs.thePropertySlot )
    {
      ;
    }

    // copy constructor 

    PropertySlotProxy( PropertySlotProxyCref rhs )
      :
      thePropertySlot( rhs.thePropertySlot )
    {
      ;
    }

    bool operator!=( PropertySlotProxyCref rhs ) const
    {
      if( rhs.thePropertySlot != this->thePropertySlot )
	{
	  return true;
	}
      return false;
    }

      
  private:
      
    PropertySlotProxy( void );
      
  private:    

    PropertySlotPtr   thePropertySlot;
    
    
  };
#endif // 0

  /** @} */ //end of libecs_module 

}


#endif /* __PROPERTYSLOT_HPP */
