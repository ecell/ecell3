//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2006-2009 Keio University
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
// written by Satya Arjunan <satya.arjunan@gmail.com>
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//

#include <SpatiocyteTauLeapProcess.hpp>
#include <SpatiocyteSpecies.hpp>

LIBECS_DM_INIT(SpatiocyteTauLeapProcess, Process); 

void SpatiocyteTauLeapProcess::set_g(unsigned& currHOR, RealMethod& g_method,
                                     const int v)
{
  unsigned aHOR(0);
  RealMethod aG;
  if(theOrder == 1)
    {
      aG = &SpatiocyteTauLeapProcess::g_order_1;
      aHOR = 1;
    }
  else if(theOrder == 2)
    {
      if(v == -1)
        {
          aG = &SpatiocyteTauLeapProcess::g_order_2_1;
          aHOR = 21;
        }
      else
        {
          aG = &SpatiocyteTauLeapProcess::g_order_2_2;
          aHOR = 22;
        }
    }
  else if(theOrder == 3)
    {
      if(v == -1)
        {
          aG = &SpatiocyteTauLeapProcess::g_order_3_1;
          aHOR = 31;
        }
      else if(v == -2)
        {
          aG = &SpatiocyteTauLeapProcess::g_order_3_2;
          aHOR = 32;
        }
      else
        {
          aG = &SpatiocyteTauLeapProcess::g_order_3_3;
          aHOR = 33;
        }
    }
  if(aHOR > currHOR)
    {
      currHOR = aHOR;
      g_method = aG;
    }
}
