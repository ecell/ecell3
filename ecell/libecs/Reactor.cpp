//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2000 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-CELL is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-CELL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-CELL -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Kouichi Takahashi <shafi@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//

#include "Util.hpp"
#include "Reactant.hpp"
#include "RootSystem.hpp"
#include "Stepper.hpp"
#include "FullID.hpp"
#include "Substance.hpp"

#include "Reactor.hpp"


namespace libecs
{

  void Reactor::makeSlots()
  {
    //FIXME: get methods
    createPropertySlot( "AppendSubstrate",*this,&Reactor::setAppendSubstrate,
			NULLPTR );
    createPropertySlot( "AppendProduct",*this,&Reactor::setAppendProduct,
			NULLPTR );
    createPropertySlot( "AppendCatalyst",*this,&Reactor::setAppendCatalyst,
			NULLPTR );
    createPropertySlot( "AppendEffector",*this,&Reactor::setAppendEffector,
			NULLPTR );

    createPropertySlot( "SubstrateList",*this,&Reactor::setSubstrateList,
			&Reactor::getSubstrateList);
    createPropertySlot( "ProductList",*this,&Reactor::setProductList,
			&Reactor::getProductList);
    createPropertySlot( "CatalystList",*this,&Reactor::setCatalystList,
			&Reactor::getCatalystList);
    createPropertySlot( "EffectorList",*this,&Reactor::setEffectorList,
			&Reactor::getEffectorList);

    createPropertySlot( "InitialActivity",*this,&Reactor::setInitialActivity,
			&Reactor::getInitialActivity );
  }

  void Reactor::setAppendSubstrate( UVariableVectorRCPtrCref aMessage )
  {
    //FIXME: range check
    appendSubstrate( FullID( (*aMessage)[0].asString() ), (*aMessage)[1].asInt() );
  }

  void Reactor::setAppendProduct( UVariableVectorRCPtrCref aMessage )
  {
    //FIXME: range check
    appendProduct( FullID( (*aMessage)[0].asString() ), (*aMessage)[1].asInt() );
  }

  void Reactor::setAppendCatalyst( UVariableVectorRCPtrCref aMessage )
  {
    //FIXME: range check
    appendCatalyst( FullID( (*aMessage)[0].asString() ), (*aMessage)[1].asInt() );
  }

  void Reactor::setAppendEffector( UVariableVectorRCPtrCref aMessage )
  {
    //FIXME: range check
    appendEffector( FullID( (*aMessage)[0].asString() ), (*aMessage)[1].asInt() );
  }

  void Reactor::setSubstrateList( UVariableVectorRCPtrCref aMessage )
  {
    //    cerr << "not implemented yet." << endl;
  }

  void Reactor::setProductList( UVariableVectorRCPtrCref aMessage )
  {
    //    cerr << "not implemented yet." << endl;
  }

  void Reactor::setEffectorList( UVariableVectorRCPtrCref aMessage )
  {
    //    cerr << "not implemented yet." << endl;
  }

  void Reactor::setCatalystList( UVariableVectorRCPtrCref aMessage )
  {
    //    cerr << "not implemented yet." << endl;
  }

  const UVariableVectorRCPtr Reactor::getSubstrateList() const
  {
    UVariableVectorRCPtr aVectorPtr( new UVariableVector );
    aVectorPtr->reserve( theSubstrateList.size() );
  
    for( ReactantVectorConstIterator i( theSubstrateList.begin() );
	 i != theSubstrateList.end() ; ++i )
      {
	aVectorPtr->push_back( (*i)->getSubstance().getFullID().getString() );
      }

    return aVectorPtr;
  }

  const UVariableVectorRCPtr Reactor::getProductList() const
  {
    UVariableVectorRCPtr aVectorPtr( new UVariableVector );
    aVectorPtr->reserve( theProductList.size() );
  
    for( ReactantVectorConstIterator i( theProductList.begin() );
	 i != theProductList.end() ; ++i )
      {
	aVectorPtr->push_back( (*i)->getSubstance().getFullID().getString() );
      }

    return aVectorPtr;
  }

  const UVariableVectorRCPtr Reactor::getEffectorList() const
  {
    UVariableVectorRCPtr aVectorPtr( new UVariableVector );
    aVectorPtr->reserve( theEffectorList.size() );
  
    for( ReactantVectorConstIterator i( theEffectorList.begin() );
	 i != theEffectorList.end() ; ++i )
      {
	aVectorPtr->push_back( (*i)->getSubstance().getFullID().getString() );
      }

    return aVectorPtr;
  }

  const UVariableVectorRCPtr Reactor::getCatalystList() const
  {
    UVariableVectorRCPtr aVectorPtr( new UVariableVector );
    aVectorPtr->reserve( theCatalystList.size() );
  
    for( ReactantVectorConstIterator i( theCatalystList.begin() );
	 i != theCatalystList.end() ; ++i )
      {
	aVectorPtr->push_back( (*i)->getSubstance().getFullID().getString() );
      }

    return aVectorPtr;
  }

  void Reactor::appendSubstrate( FullIDCref aFullID, IntCref aCoefficient )
  {
    SystemPtr aRootSystem( getRootSystem() );
    SystemPtr aSystem( aRootSystem->getSystem( aFullID.getSystemPath() ) );
    SubstancePtr aSubstance( aSystem->getSubstance( aFullID.getID() ) );

    appendSubstrate( *aSubstance, aCoefficient );
  }

  void Reactor::appendProduct( FullIDCref aFullID, IntCref aCoefficient )
  {
    SystemPtr aRootSystem( getRootSystem() );
    SystemPtr aSystem( aRootSystem->getSystem( aFullID.getSystemPath() ) );
    SubstancePtr aSubstance( aSystem->getSubstance( aFullID.getID() ) );
  
    appendProduct( *aSubstance, aCoefficient );
  }

  void Reactor::appendCatalyst( FullIDCref aFullID, IntCref aCoefficient)
  {
    SystemPtr aRootSystem( getRootSystem() );
    SystemPtr aSystem( aRootSystem->getSystem( aFullID.getSystemPath() ) );
    SubstancePtr aSubstance( aSystem->getSubstance( aFullID.getID() ) );
  
    appendCatalyst( *aSubstance, aCoefficient );
  }

  void Reactor::appendEffector( FullIDCref aFullID, IntCref aCoefficient )
  {
    SystemPtr aRootSystem( getRootSystem() );
    SystemPtr aSystem( aRootSystem->getSystem( aFullID.getSystemPath() ) );
    SubstancePtr aSubstance( aSystem->getSubstance( aFullID.getID() ) );
  
    appendEffector( *aSubstance, aCoefficient );
  }

  void Reactor::setInitialActivity( RealCref anActivity )
  {
    theInitialActivity = anActivity;

    theActivity = theInitialActivity * 
      getSuperSystem()->getStepper()->getStepInterval();
  }

  Reactor::Reactor() 
    :
    theInitialActivity( 0 ),
    theActivityBuffer( 0 ),
    theActivity( 0 )
  {
    makeSlots();
  }

  Reactor::~Reactor()
  {
    // delete all Reactants
    for( ReactantVectorConstIterator i( theSubstrateList.begin() );
	 i != theSubstrateList.end() ; ++i )
      {
	delete *i;
      }
    for( ReactantVectorConstIterator i( theProductList.begin() );
	 i != theProductList.end() ; ++i )
      {
	delete *i;
      }
    for( ReactantVectorConstIterator i( theCatalystList.begin() );
	 i != theCatalystList.end() ; ++i )
      {
	delete *i;
      }
    for( ReactantVectorConstIterator i( theEffectorList.begin() );
	 i != theEffectorList.end() ; ++i )
      {
	delete *i;
      }
  }


  void Reactor::appendSubstrate( SubstanceRef aSubstrate, 
				 IntCref aCoefficient )
  {
    ReactantPtr aReactantPtr( new Reactant( aSubstrate, aCoefficient ) );
    theSubstrateList.push_back( aReactantPtr );
  }

  void Reactor::appendProduct( SubstanceRef aProduct, IntCref aCoefficient )
  {
    ReactantPtr aReactantPtr( new Reactant( aProduct, aCoefficient ) );
    theProductList.push_back( aReactantPtr );
  }

  void Reactor::appendCatalyst( SubstanceRef aCatalyst, IntCref aCoefficient )
  {
    ReactantPtr aReactantPtr( new Reactant( aCatalyst, aCoefficient ) );
    theCatalystList.push_back( aReactantPtr );
  }

  void Reactor::appendEffector( SubstanceRef anEffector, IntCref aCoefficient )
  {
    ReactantPtr aReactantPtr( new Reactant( anEffector, aCoefficient ) );
    theEffectorList.push_back( aReactantPtr );
  }


  void Reactor::initialize()
  {
    ; // do nothing
  }


} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
