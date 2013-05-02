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

#include <MicroscopyTrackingProcess.hpp>

LIBECS_DM_INIT(MicroscopyTrackingProcess, Process); 

void MicroscopyTrackingProcess::incSpeciesLatticeCount()
{
  for(unsigned int i(0); i != thePositiveSpecies.size(); ++i)
    {
      Species* aSpecies(thePositiveSpecies[i]);
      aSpecies->updateMolecules();
      unsigned int aMoleculeSize(aSpecies->size());
      for(unsigned int j(0); j != aMoleculeSize; ++j)
        { 
          unsigned int aCoord(aSpecies->getCoord(j));
          Voxel& aMolecule((*theLattice)[aCoord]);
          ++theFreqLattice[theLatticeSpeciesIndices[i][0]][aCoord];
          for(unsigned int k(1); k != theLatticeSpeciesIndices[i].size(); ++k)
            {
              ++theFreqLattice[theLatticeSpeciesIndices[i][k]][
                aMolecule.adjoiningCoords[k-1]];
            }
        }
    }
}

void MicroscopyTrackingProcess::logFluorescentSpecies()
{
  std::vector<Point> aPoints;
  std::vector<unsigned int> aCoords;
  for(unsigned int i(0); i != theFreqLatticeSize; ++i)
    { 
      for(unsigned int j(0); j != theLatticeSpecies.size(); ++j)
        {
          if(theFreqLattice[j][i])
            {
              aCoords.push_back(i);
              /*
              if(theLatticeSpecies[j]->getIsPolymer())
                {
                  aPoints.push_back((*theLattice)[i].subunit->subunitPoint);
                }
              else if((*theLattice)[i].point)
                */
              if((*theLattice)[i].point)
                {
                  aPoints.push_back(*(*theLattice)[i].point);
                }
              else
                {
                  aPoints.push_back(theSpatiocyteStepper->coord2point(i));
                }
              break;
            }
        }
    }
  double aCurrentTime(theSpatiocyteStepper->getCurrentTime());
  theLogFile.write((char*)(&aCurrentTime), sizeof(aCurrentTime));
  unsigned int pointSize(aPoints.size());
  theLogFile.write((char*)(&pointSize), sizeof(pointSize));
  for(unsigned int i(0); i != pointSize; ++i)
    {
      Point aPoint(aPoints[i]);
      theLogFile.write((char*)(&aPoint), sizeof(aPoint));
    }
  for(unsigned int i(0); i != theLatticeSpecies.size(); ++i)
    {
      for(unsigned int j(0); j != pointSize; ++j)
        {
          unsigned int frequency(theFreqLattice[i][aCoords[j]]);
          theLogFile.write((char*)(&frequency), sizeof(frequency));
        }
    }
}

