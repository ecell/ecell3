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

#include <HistogramLogProcess.hpp>

void HistogramLogProcess::initializeLastOnce()
{
  theTotalIterations = Iterations;
  Origin = theSpatiocyteStepper->system2Comp(getSuperSystem())->centerPoint;
  VoxelDiameter = theSpatiocyteStepper->getVoxelRadius()*2;
  nLength = Length/VoxelDiameter;
  nRadius = Radius/VoxelDiameter;
  binInterval = nLength/(Bins+1);
  initializeVectors();
}

void HistogramLogProcess::saveFileData(std::ofstream& aFile,
                                       const unsigned iterations)
{
  double aTime(LogStart);
  for(unsigned i(0); i != theLogValues.size(); ++i)
    {
      saveATimePoint(aFile, aTime, iterations, i);
      aTime += LogInterval;
    }
  aFile.close();
}

void HistogramLogProcess::saveATimePoint(std::ofstream& aFile,
                                          const double aTime,
                                          const unsigned anIteration,
                                          const unsigned aCnt)
{
  for(unsigned i(0); i != Bins; ++i)
    {
      aFile << aTime << "," << i;
      for(unsigned j(0); j != theLogValues[aCnt][i].size(); ++j)
        {
          aFile << "," << theLogValues[aCnt][i][j]/anIteration;
        }
      aFile << std::endl << std::flush;
    }
}

void HistogramLogProcess::saveBackup()
{
  if(SaveCounts > 0 && 
     Iterations%(int)rint(theTotalIterations/SaveCounts) == 0)
    {
      std::string aFileName(FileName.c_str());
      aFileName = aFileName + ".back";
      cout << "Saving backup data in: " << aFileName << std::endl;
      std::ofstream aFile;
      aFile.open(aFileName.c_str(), std::ios::trunc);
      saveFileHeader(aFile);
      double aTime(LogStart);
      int completedIterations(theTotalIterations-Iterations);
      for(unsigned i(0); i != theLogValues.size(); ++i)
        {
          saveATimePoint(aFile, aTime, completedIterations, i);
          aTime += LogInterval;
        }
      aFile.close();
    }
}

void HistogramLogProcess::saveFileHeader(std::ofstream& aFile)
{
  if(theTotalIterations != 1)
    {
      aFile << std::setprecision(15);
    }
  aFile << "Time," << std::scientific << binInterval*VoxelDiameter;
  for(unsigned i(0); i != theProcessSpecies.size(); ++i)
    {
      aFile << "," << getIDString(theProcessSpecies[i]);
    }
  aFile << std::endl << std::flush;
}

void HistogramLogProcess::logValues()
{
  if(Iterations == theTotalIterations)
    {
      initLogValues();
    }
  if(Collision)
    {
      logCollision();
    }
  else
    {
      logDensity();
    }
}

void HistogramLogProcess::initLogValues()
{
  theLogValues.resize(timePointCnt+1);
  theLogValues[timePointCnt].resize(Bins);
  for(unsigned i(0); i != Bins; ++i)
    {
      theLogValues[timePointCnt][i].resize(theProcessSpecies.size(), 0);
    }
}

void HistogramLogProcess::logCollision()
{
  for(unsigned i(0); i != theProcessSpecies.size(); ++i)
    {
      Species* aSpecies(theProcessSpecies[i]);
      for(unsigned j(0); j != aSpecies->size(); ++j)
        {
          unsigned bin;
          if(isInside(bin, aSpecies->getPoint(j)))
            {
              theLogValues[timePointCnt][bin][i] += 
                aSpecies->getCollisionCnt(j);
            }
        }
    }
}

void HistogramLogProcess::logDensity()
{
  for(unsigned i(0); i != theProcessSpecies.size(); ++i)
    {
      Species* aSpecies(theProcessSpecies[i]);
      for(unsigned j(0); j != aSpecies->size(); ++j)
        {
          unsigned bin;
          if(isInside(bin, aSpecies->getPoint(j)))
            {
              theLogValues[timePointCnt][bin][i] += 1;
            }
        }
    }
}

void HistogramLogProcess::initializeVectors()
{
  //Minus end
  Minus.x = -nLength/2;
  Minus.y = 0;
  Minus.z = 0;
  Comp* aComp(theSpatiocyteStepper->system2Comp(getSuperSystem()));
  Point tmpOrigin;
  tmpOrigin.x = OriginX*aComp->lengthX/2;
  tmpOrigin.y = OriginY*aComp->lengthY/2;
  tmpOrigin.z = OriginZ*aComp->lengthZ/2;
  //Rotated Minus end
  theSpatiocyteStepper->rotateX(aComp->rotateX, &Minus, -1);
  theSpatiocyteStepper->rotateY(aComp->rotateY, &Minus, -1);
  theSpatiocyteStepper->rotateZ(aComp->rotateZ, &Minus, -1);
  theSpatiocyteStepper->rotateX(RotateX, &Minus, 1);
  theSpatiocyteStepper->rotateY(RotateY, &Minus, 1);
  theSpatiocyteStepper->rotateZ(RotateZ, &Minus, 1);
  theSpatiocyteStepper->rotateX(aComp->rotateX, &tmpOrigin, -1);
  theSpatiocyteStepper->rotateY(aComp->rotateY, &tmpOrigin, -1);
  theSpatiocyteStepper->rotateZ(aComp->rotateZ, &tmpOrigin, -1);
  add_(tmpOrigin, Origin);
  add_(Minus, tmpOrigin);
  //Direction vector from the Minus end to center
  //Direction vector from the Minus end to center
  lengthVector = sub(tmpOrigin, Minus);
  //Make direction vector a unit vector
  norm_(lengthVector);
}


bool HistogramLogProcess::isInside(unsigned& bin, Point N)
{
  Point D(lengthVector);
  Point E(Minus);
  double t((D.x*N.x+D.y*N.y+D.z*N.z-D.x*E.x-D.y*E.y-D.z*E.z)/
           (D.x*D.x +D.y*D.y+D.z*D.z));
  if(t < 0)
    {
      return false;
    }
  bin = (unsigned)floor(t/binInterval);
  if(bin >= Bins)
    {
      return false;
    }
  double dist(sqrt(pow(-N.x+E.x+D.x*t, 2)+pow(-N.y+E.y+D.y*t, 2)+
                   pow(-N.z+E.z+D.z*t, 2)));
  if(dist > nRadius)
    {
      return false;
    }
  return true;
}

LIBECS_DM_INIT_STATIC(HistogramLogProcess, Process); 
