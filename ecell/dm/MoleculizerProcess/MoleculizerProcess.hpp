//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2007 Keio University
//       Copyright (C) 2005-2007 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell System is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell System is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell System -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
// Contact information:
//   Nathan Addy, Research Associate     Voice: 510-981-8748
//   The Molecular Sciences Institute    Email: addy@molsci.org  
//   2168 Shattuck Ave.                  
//   Berkeley, CA 94704
//
//END_HEADER

#ifndef __MOLECULIZERPROCESS_HPP
#define __MOLECULIZERPROCESS_HPP

#include <libecs/libecs.hpp>
#include <libecs/Process.hpp>
#include <libecs/Stepper.hpp>
#include <libecs/FullID.hpp>

#include "mzr/moleculizer.hh"
#include "RegisterClass.hpp"

#include <functional>

USE_LIBECS;

/**************************************************************
       MProcess
**************************************************************/

namespace mzr
{
  class moleculizer;
}

LIBECS_DM_CLASS( MProcess, Process )
{

 public:
  LIBECS_DM_OBJECT( MProcess, Process)
    {

      INHERIT_PROPERTIES( Process );

      // These are the important ones....  
      PROPERTYSLOT_SET_GET( String, ModelFile);
      // PROPERTYSLOT_SET_GET( String, Model);
      
      // This is the stepper that the newly created processes should be added to.
      PROPERTYSLOT_SET_GET( String, GillespieProcessStepperID);

      // PROPERTYSLOT_SET_GET( Integer, NetworkExpansionDepth);
    }

  MProcess();
  ~MProcess();

  virtual void initialize();
  virtual void fire();


  SIMPLE_SET_GET_METHOD( String, ModelFile);
  SIMPLE_SET_GET_METHOD( String, GillespieProcessStepperID );

 private:
  void updateSpeciesChanges();
  void createSpeciesAndReactions(bool initPopulationToZero = true);


  typedef mzr::generatedDifference::newSpeciesEntry newSpeciesEntry;
  typedef mzr::generatedDifference::newReactionEntry newReactionEntry;
  typedef mzr::generatedDifference::key SpeciesKey;
  typedef mzr::generatedDifference::key ReactionKey;

  void initializeMoleculizerObject();
  void createNewSpecies(const newSpeciesEntry& newSpecies, bool initPopulationToZero);
  void createNewReaction(const newReactionEntry& newRxn);

  String ModelFile;
  String GillespieProcessStepperID;
  mzr::moleculizer* moleculizerObject;
  Integer rxnNdx;
  ModelPtr ptrModel;
  SystemPath compartmentName;
  SystemPtr compartmentPtr;
  RegisterClass* theRegisterClass;

 private:
  class addSubstratesToRxn : public std::unary_function<std::pair<mzr::species*, int>, void>
  {
  public:
    addSubstratesToRxn( ProcessPtr newRxnPtr, SystemPtr containingSystemPtr )
      :
      rxnProcessPtr( newRxnPtr ),
      parentSystemPtr( containingSystemPtr ),
      substrateNdx( 0 )
    {}

    void operator()(std::pair< mzr::species*, int> aSubstrate);

  private:
    ProcessPtr rxnProcessPtr;
    SystemPtr parentSystemPtr;
    unsigned int substrateNdx;

  private:
    static const String alphabet;
  };

  class addProductsToRxn : public std::unary_function<std::pair<mzr::species*, int>, void>
  {
  public:
    addProductsToRxn(ProcessPtr reactionPtr, SystemPtr containingSystemPtr, MProcess& aMoleculizerProcess)
      :
      rxnProcessPtr( reactionPtr ),
      parentSystemPtr( containingSystemPtr ),
      parentProcess( aMoleculizerProcess),
      productNdx( 0 )
    {}

    void operator()( std::pair<mzr::species*, int> aProduct);

  private:
    ProcessPtr rxnProcessPtr;
    SystemPtr parentSystemPtr;
    MProcess& parentProcess;
    unsigned int productNdx;

  private:
    static const String alphabet;
  };

};








#endif
