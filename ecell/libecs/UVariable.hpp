//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2001 Keio University
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

  DECLARE_CLASS( UConstantData );
  DECLARE_CLASS( UConstantStringData );
  DECLARE_CLASS( UConstantRealData );
  DECLARE_CLASS( UConstantIntData );



  class UConstantData
  {

  public:

    virtual ~UConstantData()
    {
      ; // do nothing
    }

    virtual const String asString() const = 0;
    virtual const Real   asReal()  const = 0;
    virtual const Int    asInt()    const = 0;

    virtual UConstantDataPtr createClone() const = 0;

  protected:
  
    UConstantData( UConstantDataCref ) {}
    UConstantData() {}

  private:

    UConstantCref operator= ( UConstantCref );

  };


  class UConstantStringData : public UConstantData
  {
  
  public:

    UConstantStringData( StringCref  str ) 
      : 
      theString( str ) 
    {
      ; // do nothing
    }
  
    UConstantStringData( const Real f );
    UConstantStringData( const Int   i );

    UConstantStringData( UConstantDataCref uvi )
      :
      theString( uvi.asString() )
    {
      ; // do nothing
    }

    virtual const String asString() const { return theString; }
    virtual const Real  asReal() const;
    virtual const Int    asInt() const;

    virtual UConstantDataPtr createClone() const
    {
      return new UConstantStringData( *this );
    }

  private:

    String theString;

  };

  class UConstantRealData : public UConstantData
  {

  public:

    UConstantRealData( StringCref str );
    UConstantRealData( const Real      f ) 
      : 
      theReal( f ) 
    {
      ; // do nothing
    }

    UConstantRealData( const Int        i ) 
      : 
      theReal( static_cast<Real>( i ) )
    {
      ; // do nothing
    }

    virtual const String asString() const;
    virtual const Real  asReal() const { return theReal; }
    // FIXME: range check
    virtual const Int    asInt() const { return static_cast<Int>( theReal ); }

    virtual UConstantDataPtr createClone() const
    {
      return new UConstantRealData( *this );
    }

  private:

    Real theReal;

  };

  class UConstantIntData : public UConstantData
  {

  public:

    UConstantIntData( StringCref str );
    UConstantIntData( const Real      f );
    UConstantIntData( const Int        i ) 
      : 
      theInt( i ) 
    {
      ; // do nothing
    }

    virtual const String asString() const;
    virtual const Real   asReal() const { return static_cast<Real>( theInt ); }
    virtual const Int    asInt() const   { return theInt; }
  
    virtual UConstantDataPtr createClone() const
    {
      return new UConstantIntData( *this );
    }

  private:

    Int theInt;

  };

  class UConstantNoneData : public UConstantData
  {

  public: 

    UConstantNoneData() {}


    virtual const String asString() const 
    { 
      static String aNoneString;
      return aNoneString;
    }
    virtual const Real   asReal() const   { return 0.0; }
    virtual const Int    asInt() const    { return 0; }
  
    virtual UConstantDataPtr createClone() const
    {
      return new UConstantNoneData;
    }

  };



  class UConstant
  {

  public:

    enum Type
      {
	NONE   = 0,
	REAL   = 1,
	INT    = 2,
	STRING = 3
      };

  
    UConstant()
      :
      theData( new UConstantNoneData ),
      theType( NONE )
    {
      ; // do nothing
    }

    UConstant( StringCref  string ) 
      :
      theData( new UConstantStringData( string ) ),
      theType( STRING )
    {
      ; // do nothing
    }
  
    UConstant( const Real f )      
      :
      theData( new UConstantRealData( f ) ),
      theType( REAL )
    {
      ; // do nothing
    }

    UConstant( const Int   i )      
      :
      theData( new UConstantIntData( i ) ),
      theType( INT )
    {
      ; // do nothing
    }

    UConstant( UConstantCref uv )
      :
      theData( uv.createDataClone() ),
      theType( uv.getType() )
    {
      ; // do nothing
    }

    virtual ~UConstant()
    {
      delete theData;
    }

    UConstantCref operator= ( UConstantCref rhs )
    {
      if( this != &rhs )
	{
	  delete theData;
	  theData = rhs.createDataClone();
	  theType = rhs.getType();
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

    const Type getType() const
    {
      return theType;
    }

  protected:

    UConstantDataPtr createDataClone() const
    {
      return theData->createClone();
    }

  private:

    UConstantDataPtr theData;
    Type             theType;

  };


} // namespace libecs

#endif /* ___UNIVERSALVARIABLE_H___ */
