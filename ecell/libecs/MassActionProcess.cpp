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

#include "MassActionProcess.hpp"

LIBECS_DM_INIT(MassActionProcess, Process); 

void MassActionProcess::fire()
{ 
  if(!theSpace)
    {
      for(VariableReferenceVector::iterator
          i(theVariableReferenceVector.begin());
          i != theVariableReferenceVector.end(); ++i)
        {
          Variable* aVariable((*i).getVariable()); 
          theSpace = 
            aVariable->getSuperSystem()->getSizeVariable()->getValue()*1e-3;
          if(theSpace)
            {
              break;
            }
        }
    }
  double velocity(k);
  velocity *= theSpace;
  for(VariableReferenceVector::iterator
      s(theVariableReferenceVector.begin());
      s != theZeroVariableReferenceIterator; ++s)
    {
      VariableReference aVariableReference(*s);
      Integer aCoefficient(aVariableReference.getCoefficient());
      do
        {
          ++aCoefficient;
          velocity *= aVariableReference.getVariable()->getValue()/theSpace;
        }
      while(aCoefficient != 0); 
      if(velocity < 0)
        {
          velocity = 0;
        }
    } 
  setFlux(velocity);
  for(VariableReferenceVector::iterator
      s(theVariableReferenceVector.begin());
      s != theZeroVariableReferenceIterator; ++s)
    {
      VariableReference aVariableReference(*s);
      if(aVariableReference.getVariable()->getValue() < 0)
        {
          aVariableReference.getVariable()->setValue(0);
        }
    } 
}

