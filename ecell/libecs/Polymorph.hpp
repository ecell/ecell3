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

#ifndef ___POLYMORPH_HPP
#define ___POLYMORPH_HPP

#include <assert.h>

#include "libecs.hpp"
#include "convertTo.hpp"
#include "Util.hpp"

namespace libecs
{

  /** @addtogroup uvariable The Polymorph.
   The Polymorph

   @ingroup libecs
   @{ 
   */ 

  /** @file */
  
  DECLARE_CLASS( PolymorphData );

  class PolymorphData
  {

  public:

    virtual ~PolymorphData();

    virtual const String asString()        const = 0;
    virtual const Real   asReal()          const = 0;
    virtual const Int    asInt()           const = 0;
    virtual const PolymorphVector asPolymorphVector() const = 0;

    template< typename T >
    const T as() const
    {
      DefaultSpecializationInhibited();
    }

    virtual PolymorphDataPtr createClone() const = 0;

  protected:
  
    PolymorphData( PolymorphDataCref ) {}
    PolymorphData() {}

  private:

    PolymorphCref operator= ( PolymorphCref );

  };


  template <>
  inline const String PolymorphData::as() const
  {
    return asString();
  }

  template <>
  inline const Real PolymorphData::as() const
  {
    return asReal();
  }

  template <>
  inline const Int PolymorphData::as() const
  {
    return asInt();
  }

  template <>
  inline const PolymorphVector PolymorphData::as() const
  {
    return asPolymorphVector();
  }



  template< typename T >
  class ConcretePolymorphData 
    : 
    public PolymorphData
  {
  
  public:

    ConcretePolymorphData( StringCref  aValue ) 
      :
      theValue( convertTo<T>( aValue ) )
    {
      ; // do nothing
    }

    ConcretePolymorphData( RealCref aValue ) 
      :
      theValue( convertTo<T>( aValue ) )
    {
      ; // do nothing
    }

    ConcretePolymorphData( IntCref  aValue )
      :
      theValue( convertTo<T>( aValue ) )
    {
      ; // do nothing
    }

    ConcretePolymorphData( PolymorphVectorCref aValue )
      :
      theValue( convertTo<T>( aValue ) )
    {
      ; // do nothing
    }

    ConcretePolymorphData( PolymorphDataCref aValue )
      :
      theValue( aValue.as<T>() )
    {
      ; // do nothing
    }

    virtual ~ConcretePolymorphData()
    {
      ; // do nothing
    }
  
    virtual const String asString() const 
    { 
      return convertTo<String>( theValue ); 
    }

    virtual const Real   asReal()  const { return convertTo<Real>( theValue );}
    virtual const Int    asInt()   const { return convertTo<Int>( theValue ); }
    virtual const PolymorphVector asPolymorphVector() const
    { return convertTo<PolymorphVector>( theValue ); }

    virtual PolymorphDataPtr createClone() const
    {
      return new ConcretePolymorphData<T>( *this );
    }

  private:

    T theValue;

  };

  class PolymorphNoneData 
    : 
    public PolymorphData
  {

  public: 

    PolymorphNoneData() {}

    virtual ~PolymorphNoneData();

    virtual const String asString() const;
    virtual const Real   asReal() const   { return 0.0; }
    virtual const Int    asInt() const    { return 0; }
    virtual const PolymorphVector asPolymorphVector() const;
  
    virtual PolymorphDataPtr createClone() const
    {
      return new PolymorphNoneData;
    }

  };



  class Polymorph
  {

  public:

    enum Type
      {
	NONE,
	REAL, 
	INT,  
	STRING,
	POLYMORPH_VECTOR
      };

  
    Polymorph()
      :
      theData( new PolymorphNoneData )
    {
      ; // do nothing
    }

    Polymorph( StringCref  aValue ) 
      :
      theData( new ConcretePolymorphData<String>( aValue ) )
    {
      ; // do nothing
    }
  
    Polymorph( RealCref aValue )      
      :
      theData( new ConcretePolymorphData<Real>( aValue ) )
    {
      ; // do nothing
    }

    Polymorph( IntCref aValue )      
      :
      theData( new ConcretePolymorphData<Int>( aValue ) )
    {
      ; // do nothing
    }

    Polymorph( PolymorphVectorCref aValue )
      :
      theData( new ConcretePolymorphData<PolymorphVector>( aValue ) )
    {
      ; // do nothing
    }

    Polymorph( PolymorphCref aValue )
      :
      theData( aValue.createDataClone() )
    {
      ; // do nothing
    }

    ~Polymorph()
    {
      delete theData;
    }

    PolymorphCref operator=( PolymorphCref rhs )
    {
      if( this != &rhs )
	{
	  delete theData;
	  theData = rhs.createDataClone();
	}
    
      return *this;
    }

    const String asString() const
    { 
      return theData->asString(); 
    }

    const Real  asReal() const
    { 
      return theData->asReal(); 
    }
  
    const Int    asInt() const
    { 
      return theData->asInt();
    }

    const PolymorphVector asPolymorphVector() const
    { 
      return theData->asPolymorphVector();
    }

    template< typename T >
    const T as() const
    {
      DefaultSpecializationInhibited();
    }

    const Type getType() const;

    void changeType( const Type aType );


    operator String() const
    {
      return asString();
    }

    operator Real() const
    {
      return asReal();
    }

    operator Int() const
    {
      return asInt();
    }

    operator PolymorphVector() const
    {
      return asPolymorphVector();
    }

  protected:

    PolymorphDataPtr createDataClone() const
    {
      return theData->createClone();
    }

  protected:

    PolymorphDataPtr theData;

  };


  template <>
  inline const String Polymorph::as() const
  {
    return asString();
  }

  template <>
  inline const Real Polymorph::as() const
  {
    return asReal();
  }

  template <>
  inline const Int Polymorph::as() const
  {
    return asInt();
  }

  template <>
  inline const PolymorphVector Polymorph::as() const
  {
    return asPolymorphVector();
  }



  //
  // nullValue() specialization for Polymorph. See Util.hpp
  //

  template<>
  inline const Polymorph nullValue()
  {
    return Polymorph();
  }




  //
  // Below are convertTo template function specializations for Polymorph class.
  // Mainly for PolymorphVector classes
  //

  // to Polymorph object


  // identity

  template<>
  inline const Polymorph convertTo( PolymorphCref aValue, 
				    Type2Type< Polymorph > )
  {
    return aValue;
  }

  // from Real

  template<>
  inline const Polymorph convertTo( RealCref aValue, 
				    Type2Type< Polymorph > )
  {
    return Polymorph( aValue );
  }



  // to PolymorphVector object

  // identity

  template<>
  inline const PolymorphVector 
  convertTo( PolymorphVectorCref aValue, 
	     Type2Type< PolymorphVector > )
  {
    return aValue;
  }

  // from Real
  template<>
  inline const PolymorphVector 
  convertTo( RealCref aValue,
	     Type2Type< PolymorphVector > )
  {
    return PolymorphVector( 1, aValue );
  }

  // from String
  template<>
  inline const PolymorphVector 
  convertTo( StringCref aValue,
	     Type2Type< PolymorphVector > )
  {
    return PolymorphVector( 1, aValue );
  }

  // from Int
  template<>
  inline const PolymorphVector 
  convertTo( IntCref aValue, Type2Type< PolymorphVector > )
  {
    return PolymorphVector( 1, aValue );
  }

  template<>
  inline const String convertTo( PolymorphVectorCref aValue,
				 Type2Type< String > )
  {
    checkSequenceSize( aValue, 1 );
    return aValue[0].asString();
  }

  template<>
  inline const Real convertTo( PolymorphVectorCref aValue, Type2Type< Real > )
  {
    checkSequenceSize( aValue, 1 );
    return aValue[0].asReal();
  }
    
  template<>
  inline const Int convertTo( PolymorphVectorCref aValue, Type2Type< Int > )
  {
    checkSequenceSize( aValue, 1 );
    return aValue[0].asInt();
  }
    


  // @} // uvariable

} // namespace libecs


#endif /* __POLYMORPH_HPP */
