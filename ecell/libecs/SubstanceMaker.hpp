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

#ifndef ___SUBSTANCEMAKER_H___
#define ___SUBSTANCEMAKER_H___
#include "Substance.hpp"
#include "dmtool/ModuleMaker.hpp"

namespace libecs
{

  /* *defgroup libecs_module The Libecs Module 
   * This is the libecs module 
   * @{ 
   */ 
  
  class SubstanceMaker 
    : 
    public SharedModuleMaker<Substance>
  {

  public:

    SubstanceMaker() { makeClassList(); }

  protected:

    virtual void makeClassList();

  };

#define NewSubstanceModule( CLASS ) NewDynamicModule( Substance, CLASS )

  /** @} */ //end of libecs_module 

} // namespace libecs


#endif /* ___SUBSTANCEMAKER_H___ */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
