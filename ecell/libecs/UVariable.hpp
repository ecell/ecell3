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


class UniversalVariable
{

public:
  
  UniversalVariable( StringCref  string ) {}
  UniversalVariable( const Float f )      {}   
  UniversalVariable( const Int   i )      {}
  UniversalVariable( const Uint  ui )     {}       


  virtual const String asString() = 0;
  virtual const Float  asFloat()  = 0;
  virtual const Int    asInt()    = 0;
  virtual const Uint   asUint()   = 0;

};


template <class T>
class UniversalVariableInstance : public UniversalVariable
{

public:

  virtual const String asString(){}
  virtual const Float  asFloat() {}
  virtual const Int    asInt()   {}  
  virtual const Uint   asUint()  {} 

};


// specialization classes

template<> 
class UniversalVariableInstance<String>
{
  
public:

  UniversalVariableInstance( StringCref  string ) 
    : 
    theString( string ) 
  {
    ; // do nothing
  }
  UniversalVariableInstance( const Float f );
  UniversalVariableInstance( const Int   i );
  UniversalVariableInstance( const Uint  ui );

  StringCref asString() { return theString; }
  const Float  asFloat();
  const Int    asInt();
  const Uint   asUint();

private:

  String theString;

};

template<> class UniversalVariableInstance<Float>
{

public:

  UniversalVariableInstance( StringCref string );
  UniversalVariableInstance( const Float      f ) 
    : 
    theFloat( f ) 
  {
    ; // do nothing
  }

  UniversalVariableInstance( const Int        i ) 
    : 
    theFloat( static_cast<Float>( i ) )
  {
    ; // do nothing
  }

  UniversalVariableInstance( const Uint       ui )
    : 
    theFloat( static_cast<Float>( ui ) )
  {
    ; // do nothing
  }

  const String asString();
  FloatCref    asFloat() { return theFloat; }
  // FIXME: range check
  const Int    asInt()   { return static_cast<Int>( theFloat ); }
  // FIXME: range check
  const Uint   asUint()  { return static_cast<Uint>( theFloat ); }

private:

  Float theFloat;

};

template<> class UniversalVariableInstance<Int>
{

public:

  UniversalVariableInstance( StringCref string );
  UniversalVariableInstance( const Float      f );
  UniversalVariableInstance( const Int        i ) 
    : 
    theInt( i ) 
  {
    ; // do nothing
  }
  UniversalVariableInstance( const Uint       ui );

  const String asString();
  const Float  asFloat()  { return static_cast<Float>( theInt ); }
  IntCref      asInt()    { return theInt; }
  // FIXME: range check
  const Uint   asUint()   { return static_cast<Uint>( theInt ); }

private:

  Int theInt;

};

template<> class UniversalVariableInstance<Uint>
{

public:

  UniversalVariableInstance( StringCref string );
  UniversalVariableInstance( const Float      f );
  UniversalVariableInstance( const Int        i );
  UniversalVariableInstance( const Uint       ui ) : theUint( ui ) {}

  const String asString();
  const Float  asFloat(){ return static_cast<Float>( theUint ); }
  // FIXME: range check
  const Int    asInt()  { return static_cast<Int>( theUint ); }
  UintCref     asUint() { return theUint; }

private:

  Uint theUint;

};


#endif /* ___UNIVERSALVARIABLE_H___ */
