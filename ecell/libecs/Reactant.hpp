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





#ifndef __REACTANT_H___
#define __REACTANT_H___
#include "ecscore/Substance.h"


class Reactant
{

protected:

  Substance& _substance;
  int _c;

public:

  Reactant(Substance& s,const int c) : _substance(s),_c(c) {}
  virtual ~Reactant() {}

  Substance& substance() const {return _substance;}
  int coefficient() const {return _c;}
  inline Float concentration() const {return _substance.concentration();}
  inline Float quantity() const {return _substance.quantity();}
  inline Float activity() const {return _substance.activity();}
  inline Float velocity() const {return _substance.velocity();}
  inline Float velocity(Float v) const {return _substance.velocity(v);}
  inline void setQuantity(Float q) const {_substance.setQuantity(q); }
};

#endif /* __REACTANT_H___ */
