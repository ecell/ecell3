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





#ifndef ___PRIMITIVE_H___
#define ___PRIMITIVE_H___
#include <string>

class Entity;
class Substance;
class CDS;
class GenomicElement;
class Reactor;
class System;



class Primitive
{
public:

  enum Type {ENTITY = 0x01,SUBSTANCE=0x02,REACTOR=0x04,
	     SYSTEM=0x08,CDS=0x10,GENOMICELEMENT=0x20,
	     PRIMITIVE_NONE = 0x00};

  union
    {
      Entity* entity;
      Substance* substance;
      Reactor* reactor;
      System* system;
//      CDS* cds;
      GenomicElement* genomicElement;
    };

  Primitive::Type type;
  
  Primitive() : entity(NULL) {type = PRIMITIVE_NONE;}
  Primitive(Entity* e) : entity(e),type(ENTITY) {}
  Primitive(Substance* s) : substance(s),type(SUBSTANCE) {}
  Primitive(Reactor* r) : reactor(r),type(REACTOR) {}
  Primitive(System* s) : system(s),type(SYSTEM) {}
//  Primitive(CDS* g) : cds(g),type(CDS) {}
  Primitive(GenomicElement* ge) : genomicElement(ge),type(GENOMICELEMENT) {}


  static const string PrimitiveTypeString(Type type);
  static Type PrimitiveType(const string& typestring);

};


#endif /* ___PRIMITIVE_H___ */
