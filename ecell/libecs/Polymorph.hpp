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

#ifndef ___UNIVERSALVARIABLE_H___
#define ___UNIVERSALVARIABLE_H___

#include <assert.h>

#include "libecs.hpp"

namespace libecs
{

  /** @addtogroup uvariable The Polymorph.
   The Polymorph

   @ingroup libecs
   @{ 
   */ 

  /** @file */
  
  DECLARE_CLASS( PolymorphData );
  DECLARE_CLASS( PolymorphStringData );
  DECLARE_CLASS( PolymorphRealData );
  DECLARE_CLASS( PolymorphIntData );



  class PolymorphData
  {

  public:

    virtual ~PolymorphData()
    {
      ; // do nothing
    }

    virtual const String asString()        const = 0;
    virtual const Real   asReal()          const = 0;
    virtual const Int    asInt()           const = 0;

    virtual PolymorphDataPtr createClone() const = 0;

  protected:
  
    PolymorphData( PolymorphDataCref ) {}
    PolymorphData() {}

  private:

    PolymorphCref operator= ( PolymorphCref );

  };


  class PolymorphStringData 
    : 
    public PolymorphData
  {
  
  public:

    PolymorphStringData( StringCref  str ) 
      : 
      theValue( str ) 
    {
      ; // do nothing
    }
  
    PolymorphStringData( RealCref f );
    PolymorphStringData( IntCref  i );

    PolymorphStringData( PolymorphDataCref uvi )
      :
      theValue( uvi.asString() )
    {
      ; // do nothing
    }

    virtual const String asString() const { return theValue; }
    virtual const Real   asReal()   const;
    virtual const Int    asInt()    const;

    virtual PolymorphDataPtr createClone() const
    {
      return new PolymorphStringData( *this );
    }

  private:

    String theValue;

  };

  class PolymorphRealData 
    : 
    public PolymorphData
  {

  public:

    PolymorphRealData( StringCref str );
    PolymorphRealData( RealCref   f ) 
      : 
      theValue( f ) 
    {
      ; // do nothing
    }

    PolymorphRealData( IntCref    i ) 
      : 
      theValue( static_cast<Real>( i ) )
    {
      ; // do nothing
    }

    virtual const String asString() const;
    virtual const Real   asReal() const    { return theValue; }
    virtual const Int    asInt() const;

    virtual PolymorphDataPtr createClone() const
    {
      return new PolymorphRealData( *this );
    }

  private:

    Real theValue;

  };

  class PolymorphIntData 
    : 
    public PolymorphData
  {

  public:

    PolymorphIntData( StringCref str );
    PolymorphIntData( RealCref   f );
    PolymorphIntData( IntCref    i ) 
      : 
      theValue( i ) 
    {
      ; // do nothing
    }

    virtual const String asString() const;
    virtual const Real   asReal() const;
    virtual const Int    asInt()  const { return theValue; }
  
    virtual PolymorphDataPtr createClone() const
    {
      return new PolymorphIntData( *this );
    }

  private:

    Int theValue;

  };

  class PolymorphNoneData 
    : 
    public PolymorphData
  {

  public: 

    PolymorphNoneData() {}


    virtual const String asString() const 
    { 
      static String aNoneString;
      return aNoneString;
    }
    virtual const Real   asReal() const   { return 0.0; }
    virtual const Int    asInt() const    { return 0; }
  
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
	STRING
      };

  
    Polymorph()
      :
      theData( new PolymorphNoneData )
    {
      ; // do nothing
    }

    Polymorph( StringCref  string ) 
      :
      theData( new PolymorphStringData( string ) )
    {
      ; // do nothing
    }
  
    Polymorph( RealCref f )      
      :
      theData( new PolymorphRealData( f ) )
    {
      ; // do nothing
    }

    Polymorph( IntCref  i )      
      :
      theData( new PolymorphIntData( i ) )
    {
      ; // do nothing
    }

    Polymorph( PolymorphCref uv )
      :
      theData( uv.createDataClone() )
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

    const Type getType() const;

  protected:

    PolymorphDataPtr createDataClone() const
    {
      return theData->createClone();
    }

  protected:

    PolymorphDataPtr theData;

  };

  // @} // uvariable

} // namespace libecs

#endif /* ___UNIVERSALVARIABLE_H___ */
