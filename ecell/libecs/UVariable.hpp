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

  /** @addtogroup uvariable The UVariable.
   The UVariable

   @ingroup libecs
   @{ 
   */ 

  /** @file */
  
  DECLARE_CLASS( UVariableData );
  DECLARE_CLASS( UVariableStringData );
  DECLARE_CLASS( UVariableRealData );
  DECLARE_CLASS( UVariableIntData );



  class UVariableData
  {

  public:

    virtual ~UVariableData()
    {
      ; // do nothing
    }

    virtual const String asString() const = 0;
    virtual const Real   asReal()  const = 0;
    virtual const Int    asInt()    const = 0;

    virtual UVariableDataPtr createClone() const = 0;

  protected:
  
    UVariableData( UVariableDataCref ) {}
    UVariableData() {}

  private:

    UVariableCref operator= ( UVariableCref );

  };


  class UVariableStringData : public UVariableData
  {
  
  public:

    UVariableStringData( StringCref  str ) 
      : 
      theString( str ) 
    {
      ; // do nothing
    }
  
    UVariableStringData( const Real f );
    UVariableStringData( const Int   i );

    UVariableStringData( UVariableDataCref uvi )
      :
      theString( uvi.asString() )
    {
      ; // do nothing
    }

    virtual const String asString() const { return theString; }
    virtual const Real  asReal() const;
    virtual const Int    asInt() const;

    virtual UVariableDataPtr createClone() const
    {
      return new UVariableStringData( *this );
    }

  private:

    String theString;

  };

  class UVariableRealData : public UVariableData
  {

  public:

    UVariableRealData( StringCref str );
    UVariableRealData( const Real      f ) 
      : 
      theReal( f ) 
    {
      ; // do nothing
    }

    UVariableRealData( const Int        i ) 
      : 
      theReal( static_cast<Real>( i ) )
    {
      ; // do nothing
    }

    virtual const String asString() const;
    virtual const Real  asReal() const { return theReal; }
    // FIXME: range check
    virtual const Int    asInt() const { return static_cast<Int>( theReal ); }

    virtual UVariableDataPtr createClone() const
    {
      return new UVariableRealData( *this );
    }

  private:

    Real theReal;

  };

  class UVariableIntData : public UVariableData
  {

  public:

    UVariableIntData( StringCref str );
    UVariableIntData( const Real      f );
    UVariableIntData( const Int        i ) 
      : 
      theInt( i ) 
    {
      ; // do nothing
    }

    virtual const String asString() const;
    virtual const Real   asReal() const { return static_cast<Real>( theInt ); }
    virtual const Int    asInt() const   { return theInt; }
  
    virtual UVariableDataPtr createClone() const
    {
      return new UVariableIntData( *this );
    }

  private:

    Int theInt;

  };

  class UVariableNoneData : public UVariableData
  {

  public: 

    UVariableNoneData() {}


    virtual const String asString() const 
    { 
      static String aNoneString;
      return aNoneString;
    }
    virtual const Real   asReal() const   { return 0.0; }
    virtual const Int    asInt() const    { return 0; }
  
    virtual UVariableDataPtr createClone() const
    {
      return new UVariableNoneData;
    }

  };



  class UVariable
  {

  public:

    enum Type
      {
	NONE   = 0,
	REAL   = 1,
	INT    = 2,
	STRING = 3
      };

  
    UVariable()
      :
      theData( new UVariableNoneData )
    {
      ; // do nothing
    }

    UVariable( StringCref  string ) 
      :
      theData( new UVariableStringData( string ) )
    {
      ; // do nothing
    }
  
    UVariable( const Real f )      
      :
      theData( new UVariableRealData( f ) )
    {
      ; // do nothing
    }

    UVariable( const Int   i )      
      :
      theData( new UVariableIntData( i ) )
    {
      ; // do nothing
    }

    UVariable( UVariableCref uv )
      :
      theData( uv.createDataClone() )
    {
      ; // do nothing
    }

    //    virtual ~UVariable()
    ~UVariable()
    {
      delete theData;
    }

    UVariableCref operator=( UVariableCref rhs )
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

    UVariableDataPtr createDataClone() const
    {
      return theData->createClone();
    }

  protected:

    UVariableDataPtr theData;

  };

  // @} // uvariable

} // namespace libecs

#endif /* ___UNIVERSALVARIABLE_H___ */
