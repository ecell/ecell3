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

#include "OscillationAnalysisProcess.hpp"

LIBECS_DM_INIT(OscillationAnalysisProcess, Process); 

void OscillationAnalysisProcess::testMembraneBinding()
{
  if(minD->size() < 100)
    {
      Status = 1;
    }
  std::cout << "theTime:" << theTime << " Status:" << Status << std::endl;
}  

void OscillationAnalysisProcess::testLocalization(int aStatus)
{
  int quad1(0);
  int quad2(0);
  int quad3(0);
  int quad4(0);
  int aSize(minD_m->size());
  for(int i(0); i != aSize; ++i)
    {
      Point aPoint(theSpatiocyteStepper->coord2point(minD_m->getCoord(i)));
      if(aPoint.x < theCenterPoint.x/2)
        {
          ++quad1;
        }
      else if(aPoint.x >= theCenterPoint.x/2 && aPoint.x < theCenterPoint.x)
        {
          ++quad2;
        }
      else if(aPoint.x >= theCenterPoint.x && aPoint.x < theCenterPoint.x*1.5)
        {
          ++quad3;
        }
      else
        {
          ++quad4;
        }
    }
  double threshold(0.5*aSize/4);
  std::cout << "thresh:" << threshold << " size:" << aSize << std::endl;
  if(quad1 < threshold || quad2 < threshold || quad3 < threshold ||
     quad4 < threshold)
    {
      Status = aStatus;
    }
  std::cout << "theTime:" << theTime << " quad1:" << quad1 << " quad2:" << quad2
    << " quad3:" << quad3 << " quad4:" << quad4 <<  " status:" <<
    Status << std::endl;
} 

void OscillationAnalysisProcess::testOscillation()
{
  int aSize(minD_m->size());
  std::vector<double> leftPositions;
  std::vector<double> rightPositions;
  for(int i(0); i != aSize; ++i)
    {
      Point aPoint(theSpatiocyteStepper->coord2point(minD_m->getCoord(i)));
      if(aPoint.x < theCenterPoint.x)
        {
          leftPositions.push_back(aPoint.x);
        }
      else
        {
          rightPositions.push_back(aPoint.x);
        }
    }
  if(leftPositions.size() > 100)
    {
      isLeftNucleateExited = true;
    }
  if(rightPositions.size() > 100)
    {
      isRightNucleateExited = true;
    }
  if(isLeftNucleateExited && leftPositions.size() <= 80 &&
     leftPositions.size() >= 10 && rightPositions.size() > 400 &&
     leftPositions.size() > prevLeftSize)
    {
      int moleculeCnt(0);
      for(unsigned int i(0); i != leftPositions.size(); ++i)
        {
          if(leftPositions[i] <= 0.5*theCenterPoint.x)
            {
              ++moleculeCnt;
            }
        }
      std::cout << "moleculeCnt:" << moleculeCnt << std::endl;
      int currStatus(4);
      if(moleculeCnt < 0.5*leftPositions.size())
        {
          currStatus = 5; //Aberrant polar nucleation
        }
      if(isLeftNucleated)
        {
          Period = theTime - theLeftBeginTime;
          ++theCycleCount;
          theTotalPeriod += Period;
          if(prevLeftStatus == 4 && currStatus == 4 && prevRightStatus == 4)
            {
              Status = 4; //Normal nucleations after the last complete cycle
            }
          else
            {
              Status = 5; //Aberrant polar nucleation
            }
        }
      theLeftBeginTime = theTime;
      isLeftNucleated = true;
      isLeftNucleateExited = false;
      prevLeftStatus = currStatus;
    }
  else if(isRightNucleateExited && rightPositions.size() <= 80 &&
          rightPositions.size() >= 10 && leftPositions.size() > 400 &&
          rightPositions.size() > prevRightSize)
    {
      int moleculeCnt(0);
      for(unsigned int i(0); i != rightPositions.size(); ++i)
        {
          if(rightPositions[i] >= (theCenterPoint.x+0.5*theCenterPoint.x))
            {
              ++moleculeCnt;
            }
        }
      std::cout << "moleculeCnt:" << moleculeCnt << std::endl;
      int currStatus(4);
      if(moleculeCnt < 0.5*rightPositions.size())
        {
          currStatus = 5; //Aberrant polar nucleation
        }
      if(isRightNucleated)
        {
          if(prevRightStatus == 4 && currStatus == 4 && prevLeftStatus == 4)
            {
              Status = 4; //Normal nucleations after the last complete cycle
            }
          else
            {
              Status = 5; //Aberrant polar nucleation
            }
        }
      theRightBeginTime = theTime;
      isRightNucleated = true;
      isRightNucleateExited = false;
      prevRightStatus = currStatus;
    }
  prevLeftSize = leftPositions.size();
  prevRightSize = rightPositions.size();
  std::cout << "theTime:" << theTime << " left:" << leftPositions.size() << " right:" << rightPositions.size() << " current period:" << Period << " avg period:" << theTotalPeriod/theCycleCount << std::endl;
}  
