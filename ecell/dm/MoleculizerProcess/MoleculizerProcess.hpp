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
       MoleculizerProcess
**************************************************************/

namespace mzr
{
  class moleculizer;
}

LIBECS_DM_CLASS( MoleculizerProcess, Process )
{

 public:
  LIBECS_DM_OBJECT( MoleculizerProcess, Process)
    {
      INHERIT_PROPERTIES( Process );

      // 0 = no debug, 1 = verbose output to stdout.  
      PROPERTYSLOT_SET_GET( Integer, Debug);
      PROPERTYSLOT_SET_GET( Integer, RateExtrapolation);

      PROPERTYSLOT_SET_GET( String, ModelDescription);
      PROPERTYSLOT_SET_GET( String, ModelFile);

      // This is the stepper that the newly created processes should be added to.
      PROPERTYSLOT_SET_GET( String, GillespieProcessStepperID);

      PROPERTYSLOT_SET_GET( Integer, NetworkExpansionDepth);
    }

  MoleculizerProcess();
  ~MoleculizerProcess();

  virtual void initialize();
  virtual void fire();


  SIMPLE_SET_GET_METHOD( String, ModelDescription)
    SIMPLE_SET_GET_METHOD( String, ModelFile);
  SIMPLE_SET_GET_METHOD( String, GillespieProcessStepperID );
  SET_METHOD(Integer, Debug)
    {
      debugMode = (bool) value;
    }

  GET_METHOD(Integer, Debug)
    {
      if (debugMode) return 1;
      else return 0;
    }

  GET_METHOD( Integer, NetworkExpansionDepth)
    {
      // Optimally this just sets the moleculizer object directly.
      return NetworkExpansionDepth;
    }

  SET_METHOD( Integer, NetworkExpansionDepth)
    {
      // Set it in the moleculizer object as well.
      NetworkExpansionDepth = value;

      if (moleculizerObject)
        {
          moleculizerObject->setGenerateDepth( value );
        }
    }


  GET_METHOD( Integer, RateExtrapolation)
    {
      return 1;
    }

  SET_METHOD( Integer, RateExtrapolation)
    {
      // Do nothing for now.
    }

 private:

  void expandMoleculizerNetworkBySpecies();
  void createSpeciesAndReactions(bool initPopulationToZero = true);
  void initializeMoleculizerObject();
  void createNewSpecies(const String& newSpecies, bool initPopulationToZero);
  void createNewReaction(const mzr::mzrReaction* newRxn);




  bool debugMode;
  String ModelDescription;
  String ModelFile;
  Integer NetworkExpansionDepth;
  bool onceInitialized;
  bool rateExtrapolation;
             
  

  String GillespieProcessStepperID;
  mzr::moleculizer* moleculizerObject;
  Integer rxnNdx;

  ModelPtr ptrModel;

  SystemPath compartmentPath;
  SystemPtr compartmentPtr;
  RegisterClass* theRegisterClass;

 private:
  class addSubstratesToRxn : public std::unary_function<std::pair<mzr::mzrSpecies*, int>, void>
  {
  public:
    // Should this be a SystemPtr...?
    addSubstratesToRxn( ProcessPtr newRxnPtr, SystemPtr containingSystemPtr )
      :
      rxnProcessPtr( newRxnPtr ),
      parentSystemPtr( containingSystemPtr ),
      substrateNdx( 0 )
    {}

    void operator()(std::pair< mzr::mzrSpecies*, int> aSubstrate);

  private:
    ProcessPtr rxnProcessPtr;
    SystemPtr parentSystemPtr;
    unsigned int substrateNdx;

  private:
    static const String alphabet;
  };

  class addProductsToRxn : public std::unary_function<std::pair<mzr::mzrSpecies*, int>, void>
  {
  public:
    addProductsToRxn(ProcessPtr reactionPtr, SystemPtr containingSystemPtr, MoleculizerProcess& aMoleculizerProcess)
      :
      rxnProcessPtr( reactionPtr ),
      parentSystemPtr( containingSystemPtr ),
      parentProcess( aMoleculizerProcess),
      productNdx( 0 )
    {}

    void operator()( std::pair<mzr::mzrSpecies*, int> aProduct);

  private:
    ProcessPtr rxnProcessPtr;
    SystemPtr parentSystemPtr;
    MoleculizerProcess& parentProcess;
    unsigned int productNdx;

  private:
    static const String alphabet;
  };

};

#endif
