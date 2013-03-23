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


#ifndef __MOLECULEPOPULATEPROCESSINTERFACE_HPP
#define __MOLECULEPOPULATEPROCESSINTERFACE_HPP

#include "SpatiocyteCommon.hpp"

class MoleculePopulateProcessInterface
{ 
public:
  virtual ~MoleculePopulateProcessInterface() {}

  virtual void populateGaussian(Species*) = 0;
  virtual void populateUniformDense(Species*, unsigned int[], unsigned int*) = 0;
  virtual void populateUniformSparse(Species* aSpecies) = 0;
  virtual void populateUniformOnDiffusiveVacant(Species* aSpecies) = 0;
  virtual int getPriority() = 0;
};

#endif /* __MOLECULEPOPULATEPROCESSINTERFACE_HPP */
