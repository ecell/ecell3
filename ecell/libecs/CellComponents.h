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

#ifndef ___CELLCOMPONENTS_H___
#define ___CELLCOMPONENTS_H___
#include "System.h"
#include "Substance.h"
#include "Reactor.h"

class RootSystem;

#if 0
class Environment : public SSystem, public RSystem
{

public:
  Environment();
  ~Environment() {}

  virtual void initialize();
  virtual void clear();
  virtual void react();
  virtual void transit();
  virtual void postern();

  /**
    this is a macro for instantiate a CellComponent used by SystemMaker.

    @see MultiClassModuleMaker, DynamicModule
    */
  static System* instance() {return new Environment;}

  virtual const char* const className() const {return "Environment";}
};

class Monolithic : public RSystem , public SSystem 
{

public:
  Monolithic();
  virtual ~Monolithic() {}


  virtual void initialize();
  virtual void clear();
  virtual void react();
  virtual void transit();
  virtual void postern();

  /**
    this is a macro for instantiate a CellComponent used by SystemMaker.

    @see MultiClassModuleMaker, DynamicModule
    */
  static System* instance() {return new Monolithic;}

  virtual const char* const className() const {return "Monolithic";}
};



class Cytoplasm : public RSystem, public SSystem, public MetaSystem
{
friend class RootSystem;
friend class RuleInterpreter;

public:
  Cytoplasm();
  virtual ~Cytoplasm() {}

  virtual void initialize();
  virtual void clear();
  virtual void react();
  virtual void transit();
  virtual void postern();

  /**
    this is a macro for instantiate a CellComponent used by SystemMaker.

    @see MultiClassModuleMaker, DynamicModule
    */
  static System* instance() {return new Cytoplasm;}

  virtual const char* const className() const {return "Cytoplasm";}
};


class Membrane : public RSystem , public SSystem
{
friend RootSystem;
friend class RuleInterpreter;

  Float _volume;
  System* _inside;
  System* _outside;

public:

  Membrane();
  virtual ~Membrane() {}

  virtual void makeSlots();

  virtual void initialize();
  virtual void clear();
  virtual void react();
  virtual void transit();
  virtual void postern();

  System* inside() const {return _inside;}
  System* outside() const {return _outside;}

  void setInside(const Message& message); 
  void setOutside(const Message& message); 

  const Message getInside(StringCref keyword);
  const Message getOutside(StringCref keyword);

  void setInside(StringCref systemname); 
  void setOutside(StringCref systemname); 

  /**
    this is a macro for instantiate a CellComponent used by SystemMaker.

    @see MultiClassModuleMaker, DynamicModule
    */
  static System* instance() {return new Membrane;}

  virtual const char* const className() const {return "Membrane";}
};


class Cell : public RSystem , public SSystem,public MetaSystem
{
friend class RootSystem;
friend class RuleInterpreter;

public:

  Cell();
  virtual ~Cell() {}

  virtual void initialize();
  virtual void clear();
  virtual void react();
  virtual void transit();
  virtual void postern();

  /**
    this is a macro for instantiate a CellComponent used by SystemMaker.

    @see MultiClassModuleMaker, DynamicModule
    */
  static System* instance() {return new Cell;}

  virtual const char* const className() const {return "Cell";}
};

#endif // 0

#endif /* ___CELLCOMPONENTS_H___ */



