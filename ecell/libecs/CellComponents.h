//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
// 		This file is part of Serizawa (E-CELL Core System)
//
//	       written by Kouichi Takahashi  <shafi@sfc.keio.ac.jp>
//
//                              E-CELL Project,
//                          Lab. for Bioinformatics,  
//                             Keio University.
//
//             (see http://www.e-cell.org for details about E-CELL)
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// Serizawa is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// Serizawa is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with Serizawa -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER





#ifndef ___CELLCOMPONENTS_H___
#define ___CELLCOMPONENTS_H___
#include "ecscore/System.h"
#include "ecscore/Substance.h"
#include "ecscore/Reactor.h"

class RootSystem;

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

    \sa MultiClassModuleMaker, DynamicModule
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

    \sa MultiClassModuleMaker, DynamicModule
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

    \sa MultiClassModuleMaker, DynamicModule
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

  const Message getInside(const string& keyword);
  const Message getOutside(const string& keyword);

  void setInside(const string& systemname); 
  void setOutside(const string& systemname); 

  /**
    this is a macro for instantiate a CellComponent used by SystemMaker.

    \sa MultiClassModuleMaker, DynamicModule
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

    \sa MultiClassModuleMaker, DynamicModule
    */
  static System* instance() {return new Cell;}

  virtual const char* const className() const {return "Cell";}
};


#if 0  /* this class does not supported in current version */
class GXSystem : public RSystem
{
friend RootSystem;
friend class RuleInterpreter;
  
  Reactor* _activityIndex;

protected:


public:
  GXSystem();
  virtual ~GXSystem() {}

  virtual void initialize();
  virtual void clear();
  virtual void react();
  virtual void transit();

  virtual Float activity(); 

  void setIndex(Reactor* index) {_activityIndex = index;}

  /**
    this is a macro for instantiate a CellComponent used by SystemMaker.

    \sa MultiClassModuleMaker, DynamicModule
    */
  static System* instance() {return new GXSystem;}

  virtual const char* const className() const {return "GXSystem";}
};

class Chromosome : public Genome, public RSystem, public SSystem
{
friend class RootSystem;

public:
  Chromosome();
  virtual ~Chromosome() {}

  virtual void initialize();
  virtual void clear();
  virtual void react();
  virtual void transit();

  /**
    this is a macro for instantiate a CellComponent used by SystemMaker.

    \sa MultiClassModuleMaker, DynamicModule
    */
  static System* instance() {return new Chromosome;}

  virtual const char* const className() const {return "Chromosome";}
};

class GenomicElement : public SSystem
{
friend class RootSystem;
friend class RuleInterpreter;

  int _start;
  int _length;
  vector<BindingSite*> _bindingSiteList;

public:

  GenomicElement();
  virtual ~GenomicElement() {}

  int startpoint() const {return _start;}
  int length() const {return _length;}

  inline int position() const {return (int)(positionIndex()*0.5);}
  inline int positionIndex() const {return 2*_start+_length;}
  inline int endPosition() const {return _start+_length-1;}

  void newBindingSite(BindingSite* bs)
    {_bindingSiteList.insert(_bindingSiteList.end(),bs);}

  /**
    this is a macro for instantiate a CellComponent used by SystemMaker.

    \sa MultiClassModuleMaker, DynamicModule
    */
  static System* instance() {return new GenomicElement;}

  virtual const char* const className() const {return "GenomicElement";}
};

class Gene : public GenomicElement
{
  friend class RootSystem;
  friend class RuleInterpreter;

  vector<CDS*> _cdsList;

public:
  Gene();
  virtual ~Gene() {}

  // setAttributes() is defined in GenomicElement class
  
  CDS* getCDS(int i=0) const {assert(_cdsList[0]); return _cdsList[0];}
  int numCDS() {return _cdsList.size();}
  bool newCDS(CDS* cds){_cdsList.insert(_cdsList.end(),cds); return true;}

  /**
    this is a macro for instantiate a CellComponent used by SystemMaker.

    \sa MultiClassModuleMaker, DynamicModule
    */
  static System* instance() {return new Gene;}

  virtual const char* const className() const {return "Gene";}
};

#endif 0

#endif /* ___CELLCOMPONENTS_H___ */



