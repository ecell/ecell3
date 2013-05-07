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


#ifndef __SpatiocyteCommon_hpp
#define __SpatiocyteCommon_hpp

#include <iostream>
#include <iomanip>
#include <math.h>
#include <libecs/libecs.hpp>
#include <libecs/FullID.hpp>
#include <PriorityQueue.hpp>

class SpatiocyteProcessInterface;
class Species;
//struct Subunit;
typedef PriorityQueue<SpatiocyteProcessInterface*> ProcessPriorityQueue;
typedef ProcessPriorityQueue::ID ProcessID;

//Lattice type:
#define HCP_LATTICE   0
#define CUBIC_LATTICE 1

//Comp dimensions:
#define VOLUME  3
#define SURFACE 2
#define LINE    1

//Comp geometries:
#define CUBOID        0
#define ELLIPSOID     1
#define CYLINDER      2
#define ROD           3
#define PYRAMID       4
#define ERYTHROCYTE   5

//CUBOID Comp surface boundary conditions:
#define REFLECTIVE     0 
#define PERIODIC       1
#define UNIPERIODIC    2
#define REMOVE_UPPER   3
#define REMOVE_LOWER   4
#define REMOVE_BOTH    5

//The 12 adjoining voxels of a voxel in the HCP lattice:
#define NORTH    0 
#define SOUTH    1
#define NW       2 
#define SW       3
#define NE       4
#define SE       5
#define EAST     6 
#define WEST     7 
#define DORSALN  8
#define DORSALS  9
#define VENTRALN 10 
#define VENTRALS 11

//The 6 adjoining voxels of a voxel in the CUBIC lattice:
#define DORSAL   4 
#define VENTRAL  5

#define INNER     0
#define OUTER     1
#define IMMED     2
#define EXTEND    3
#define SHARED    4

//Polymerization parameters
#define LARGE_DISTANCE 50
#define MAX_MONOMER_OVERLAP 0.2
#define MAX_IMMEDIATE_DISTANCE 0.2 
#define BIG_NUMBER 1e+20

struct Point 
{
  double x;
  double y;
  double z;
};

struct Voxel
{
  unsigned idx;
  unsigned short diffuseSize;
  unsigned short adjoiningSize;
  //We use short here to maintain the size of Voxel as 128 bytes which helps
  //prefetching. Species ID:
  //Try to limit the adjoiningSize <= 6:
  unsigned int coord;
  unsigned int* adjoiningCoords;
  Point* point;
  /*
  //remove initAdjoins once MicrotubuleProcess is fixed:
  unsigned int* initAdjoins;
  Subunit* subunit;
  //Contains adjoining and extended surface voxels:
  std::vector<std::vector<unsigned int> >* surfaceCoords;
  */
};

struct Comp
{
  bool isIntersectParent;
  bool isIntersectRoot;
  unsigned dimension;
  unsigned vacantID; //remove this
  unsigned interfaceID;
  int enclosed;
  int geometry;
  int xyPlane;
  int xzPlane;
  int yzPlane;
  unsigned int minRow;
  unsigned int minCol;
  unsigned int minLayer;
  unsigned int maxRow;
  unsigned int maxCol;
  unsigned int maxLayer;
  double lengthX;
  double lengthY;
  double lengthZ;
  double originX;
  double originY;
  double originZ;
  double rotateX;
  double rotateY;
  double rotateZ;
  double specVolume;
  double specArea;
  double actualVolume;
  double actualArea;
  System* system;
  Comp* surfaceSub;
  //Even if there are many adjacent diffusive compartents, use only one single
  //common id. So there is only one common diffusive Comp:
  Comp* diffusiveComp;
  Point centerPoint;
  Species* vacantSpecies;
  std::vector<Comp*> allSubs;
  std::vector<Comp*> immediateSubs;
  std::vector<Comp*> intersectPeers;
  std::vector<Comp*> intersectLowerPeers;
  std::vector<Comp*> lineSubs;
  std::vector<Species*> species;
  std::vector<unsigned int> adjoinCount;
};

//id by default is 0, it is only has non-zero value when a GFP is explicitly
//tagged:
struct Tag
{
  unsigned origin;
  unsigned id;
  unsigned rotIndex; //rotation index
  unsigned vacantIdx;
  unsigned boundCnt;
};

struct Origin
{
  Point point;
  int row;
  int layer;
  int col;
};

/*
struct Bend
{
  double angle;
  double dcm[9];
  double sphereDcm[9];
  double cylinderDcm[9];
};

struct Subunit
{
  unsigned int bendSize;
  //point is the surface point of the voxel:
  Point surfacePoint;
  //subunitPoint is the continuous point of the subunit. 
  //subunitPoint = surfacePoint if the voxel is the origin of the polymer:
  Point subunitPoint;
  //coord is the actual coordinate of the actual voxel occupied by the
  //molecule. A shared lipid voxel's subunit also points to the voxel
  //occupied by the molecule. We need to 
  //differentiate the shared and actual voxels:
  unsigned int coord;
  Species* species;
  std::vector<unsigned int> targetCoords;
  std::vector<bool> boundBends;
  std::vector<unsigned int> sourceCoords;
  std::vector<unsigned int> sharedLipids;
  std::vector<unsigned int> tmpCoords;
  std::vector<Point> targetPoints;
  std::vector<Bend> targetBends;
  //contPoints are all the continuous points that are represented by the voxel
  //pointed by this subunit:
  //There shouldn't be duplicate contPoints
  std::vector<Point> contPoints;
  //the contPointSize is the number of times the same continuous point is used.
  //so there can be duplicates of contPoints
  std::vector<int> contPointSize;
};
*/

#endif /* __SpatiocyteCommon_hpp */
