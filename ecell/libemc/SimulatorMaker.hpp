//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//


#ifndef __SIMULATORMAKER_HPP
#define __SIMULATORMAKER_HPP
#include "Simulator.hpp"
//FIXME: should be dmtool/ModuleMaker.hpp
#include "ModuleMaker.hpp"

namespace libemc
{

  /** @defgroup libemc_module The Libemc Module 
   * This is the libemc module 
   * @{ 
   */ 
  
  class SimulatorMaker : public SharedModuleMaker<Simulator>
  {
  private:

  protected:

  public:

    SimulatorMaker();
    Simulator* make( libecs::StringCref classname );
    void install( libecs::StringCref systementry );

    virtual const char* const className() const {return "SimulatorMaker";}
  };

#define NewSimulatorModule(CLASS) NewDynamicModule(Simulator,CLASS)

  /** @} */ //end of libemc_module 

} // namespace libemc

#endif /* __SIMULATORMAKER_HPP */

