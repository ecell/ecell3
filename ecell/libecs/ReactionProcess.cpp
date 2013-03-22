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

#include "ReactionProcess.hpp"

LIBECS_DM_INIT(ReactionProcess, Process);

void ReactionProcess::calculateOrder()
{ 
  theOrder = 0;
  for(VariableReferenceVector::iterator 
      i(theSortedVariableReferences.begin());
      i != theSortedVariableReferences.end(); ++i)
    {
      const int aCoefficient((*i).getCoefficient());
      Variable* aVariable((*i).getVariable());
      if(aCoefficient < 0)
        {
          theOrder -= aCoefficient; 
          //The first reactant, A:
          if(A == NULL && variableA == NULL)
            {
              if(aVariable->getName() == "HD")
                {
                  variableA = aVariable;
                  coefficientA = aCoefficient;
                }
              else
                {
                  A = theSpatiocyteStepper->getSpecies(aVariable);
                }
            }
          //The second reactant, B:
          else
            {
              if(aVariable->getName() == "HD")
                {
                  variableB = aVariable;
                  coefficientB = aCoefficient;
                }
              else
                {
                  B = theSpatiocyteStepper->getSpecies(aVariable);
                }
            }
        }
      else if(aCoefficient > 0)
        {
          //The first product, C:
          if(C == NULL && variableC == NULL)
            {
              if(aVariable->getName() == "HD")
                {
                  variableC = aVariable;
                  coefficientC = aCoefficient;
                }
              else
                {
                  C = theSpatiocyteStepper->getSpecies(aVariable);
                }
            }
          //The second product, D:
          else if(D == NULL && variableD == NULL)
            {
              if(aVariable->getName() == "HD")
                {
                  variableD = aVariable;
                  coefficientC = aCoefficient;
                }
              else
                {
                  D = theSpatiocyteStepper->getSpecies(aVariable);
                }
            }
          //The third product, F:
          else if(F == NULL && variableF == NULL)
            {
              if(aVariable->getName() == "HD")
                {
                  variableF = aVariable;
                  coefficientF = aCoefficient;
                }
              else
                {
                  F = theSpatiocyteStepper->getSpecies(aVariable);
                }
            }
        }
      //aCoefficient == 0:
      else
        {
          if(aVariable->getName() == "HD")
            {
              variableE = aVariable;
              coefficientE = aCoefficient;
            }
          else
            {
              E = theSpatiocyteStepper->getSpecies(aVariable);
            }
        }
    }
} 

