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





#ifndef ___REACTORMAKER_H___
#define ___REACTORMAKER_H___
#include <stl.h>
#include "Koyurugi/Reactor.h"
#include "util/SharedModuleMaker.h"

class ReactorMaker : public SharedModuleMaker<Reactor>
{
private:

protected:

//  virtual void makeClassList();

public:

  ReactorMaker();
  Reactor* make(const string& classname) throw(CantInstantiate);
  void install(const string& systementry);

  virtual const char* const className() const {return "ReactorMaker";}
};

#define NewReactorModule(CLASS) NewDynamicModule(Reactor,CLASS)

#endif /* ___REACTORMAKER_H___ */
