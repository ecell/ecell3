
char const ReactorMaker_C_rcsid[] = "$Id$";
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




#include <stdlib.h>
#include "Koyurugi/ReactorMaker.h"
#include "ecell/Serizawa.h"
#include "Koyurugi/RootSystem.h"
#include "util/Message.h"
#include "Koyurugi/Defs.h"

//template string;
//template DynamicModule<Reactor,ReactorAllocatorFunc>;
//template MultiClassModuleMaker<Reactor>;



////////////////////// ReactorMaker
 

ReactorMaker::ReactorMaker()
{
  addSearchPath(REACTOR_SO_DIR);

  char* env = getenv("REACTOR_PATH");
  if(env)
    addSearchPath(env);

  addSearchPath(Serizawa::REACTOR_PATH);

  *theMessageWindow << "ReactorMaker: Reactor path = [" << searchPath()
		    << "].\n";
}

Reactor* ReactorMaker::make(const string& classname) throw(CantInstantiate)
{
  assert(allocator(classname));
  Reactor* instance = allocator(classname)();

  if(instance == NULL)
    throw CantInstantiate(__PRETTY_FUNCTION__,
			  className() + string(": failed to instantiate a ")
			  + classname + ".");
  ++_numInstance;
  return instance;
}



