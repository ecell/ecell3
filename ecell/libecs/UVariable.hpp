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

#include "libecs.hpp"

DECLARE_CLASS( UniversalVariableData );
DECLARE_CLASS( UniversalVariableStringData );
DECLARE_CLASS( UniversalVariableFloatData );
DECLARE_CLASS( UniversalVariableIntData );



class UniversalVariableData
{

public:

  virtual ~UniversalVariableData()
  {
    ; // do nothing
  }

  virtual const String asString() const = 0;
  virtual const Float  asFloat()  const = 0;
  virtual const Int    asInt()    const = 0;

  virtual const bool isString() const 
  {
    return false;
  }

  virtual const bool isFloat() const
  {
    return false;
  }

  virtual const bool isInt() const
  {
    return false;
  }

  virtual UniversalVariableDataPtr createClone() const = 0;

protected:
  
  UniversalVariableData( UniversalVariableDataCref ) {}
  UniversalVariableData() {}

private:

  UniversalVariableCref operator= ( UniversalVariableCref );

};


class UniversalVariableStringData : public UniversalVariableData
{
  
public:

  UniversalVariableStringData( StringCref  str ) 
    : 
    theString( str ) 
  {
    ; // do nothing
  }
  
  UniversalVariableStringData( const Float f );
  UniversalVariableStringData( const Int   i );

  UniversalVariableStringData( UniversalVariableDataCref uvi )
    :
    theString( uvi.asString() )
  {
    ; // do nothing
  }

  const String asString() const { return theString; }
  const Float  asFloat() const;
  const Int    asInt() const;

  virtual const bool isString() const
  {
    return true;
  }

  virtual UniversalVariableDataPtr createClone() const
  {
    return new UniversalVariableStringData( *this );
  }

private:

  String theString;

};

class UniversalVariableFloatData : public UniversalVariableData
{

public:

  UniversalVariableFloatData( StringCref str );
  UniversalVariableFloatData( const Float      f ) 
    : 
    theFloat( f ) 
  {
    ; // do nothing
  }

  UniversalVariableFloatData( const Int        i ) 
    : 
    theFloat( static_cast<Float>( i ) )
  {
    ; // do nothing
  }

  const String asString() const;
  const Float  asFloat() const { return theFloat; }
  // FIXME: range check
  const Int    asInt() const { return static_cast<Int>( theFloat ); }

  virtual const bool isFloat() const
  {
    return true;
  }

  virtual UniversalVariableDataPtr createClone() const
  {
    return new UniversalVariableFloatData( *this );
  }

private:

  Float theFloat;

};

class UniversalVariableIntData : public UniversalVariableData
{

public:

  UniversalVariableIntData( StringCref str );
  UniversalVariableIntData( const Float      f );
  UniversalVariableIntData( const Int        i ) 
    : 
    theInt( i ) 
  {
    ; // do nothing
  }

  const String asString() const;
  const Float  asFloat() const { return static_cast<Float>( theInt ); }
  const Int    asInt() const   { return theInt; }
  
  virtual const bool isInt() const
  {
    return true;
  }

  virtual UniversalVariableDataPtr createClone() const
  {
    return new UniversalVariableIntData( *this );
  }

private:

  Int theInt;

};



class UniversalVariable
{

public:
  
  UniversalVariable( StringCref  string ) 
    //    :
    //    theData( new UniversalVariableStringData( string ) )
  {
    theData = new UniversalVariableStringData( string );
    ; // do nothing
  }
  
  UniversalVariable( const Float f )      
    :
    theData( new UniversalVariableFloatData( f ) )
  {
    ; // do nothing
  }

  UniversalVariable( const Int   i )      
    :
    theData( new UniversalVariableIntData( i ) )
  {
    ; // do nothing
  }

  UniversalVariable( UniversalVariableCref uv )
    :
    theData( uv.createDataClone() )
  {
    ; // do nothing
  }

  virtual ~UniversalVariable()
  {
    delete theData;
  }

  UniversalVariableCref operator= ( UniversalVariableCref rhs )
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

  const Float  asFloat() const
  { 
    assert( theData );
    return theData->asFloat(); 
  }
  
  const Int    asInt() const
  { 
    return theData->asInt();
  }

  const bool isString() const
  { 
    return theData->isString();
  }

  const bool isFloat() const
  {
    return theData->isFloat();
  }

  const bool isInt() const
  { 
    return theData->isInt(); 
  }

protected:

  UniversalVariableDataPtr createDataClone() const
  {
    theData->createClone();
  }

private:

  UniversalVariable();

private:

  UniversalVariableDataPtr theData;

};


#endif /* ___UNIVERSALVARIABLE_H___ */
