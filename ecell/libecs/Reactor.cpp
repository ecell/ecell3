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
#include "FQPI.hpp"
#include "Reactor.hpp"

Reactor::Condition Reactor::theGlobalCondition;// = Reactor::Condition::Good;

void Reactor::makeSlots()
{
  //FIXME: get methods
  MessageSlot( "substrate",Reactor,*this,&Reactor::setSubstrate,NULL );
  MessageSlot( "product",Reactor,*this,&Reactor::setProduct,NULL );
  MessageSlot( "catalyst",Reactor,*this,&Reactor::setCatalyst,NULL );
  MessageSlot( "effector",Reactor,*this,&Reactor::setEffector,NULL );
  MessageSlot( "initialActivity",Reactor,*this,&Reactor::setInitialActivity,
	       &Reactor::getInitialActivity );
}

void Reactor::setSubstrate( MessageCref message )
{
  appendSubstrate( message.getBody( 0 ), 
		   stringTo<Int>( message.getBody( 1 ) ) );
}

void Reactor::setProduct( MessageCref message )
{
  appendProduct( message.getBody( 0 ),
		 stringTo<Int>( message.getBody( 1 ) ) );
}

void Reactor::setCatalyst( MessageCref message )
{
  appendCatalyst( message.getBody( 0 ),
		  stringTo<Int>( message.getBody( 1 ) ) );
}

void Reactor::setEffector( MessageCref message )
{
  appendEffector( message.getBody( 0 ),
		  stringTo<Int>( message.getBody( 1 ) ) );
}

void Reactor::setInitialActivity( MessageCref message )
{
  setInitialActivity( stringTo<Float>( message.getBody() ) );
}

const Message getSubstrate( StringCref )
{
  static String aKeyword( "Substrate" );
  return Message( aKeyword, "not implemented yet" );
}

const Message getProduct( StringCref )
{
  static String aKeyword( "Product" );
  return Message( aKeyword, "not implemented yet" );
}

const Message getCatalyst( StringCref )
{
  static String aKeyword( "Catalyst" );
  return Message( aKeyword, "not implemented yet" );
}

const Message getEffector( StringCref )
{
  static String aKeyword( "Effector" );
  return Message( aKeyword, "not implemented yet" );
}

const Message Reactor::getInitialActivity( StringCref keyword )
{
  return Message( keyword, theInitialActivity );
}

void Reactor::appendSubstrate( FQIDCref fqid, int coefficient )
{
  SubstancePtr aSubstance( getSuperSystem()->getRootSystem()->
			   getSubstance( fqid ) );

  appendSubstrate( *aSubstance, coefficient );
}

void Reactor::appendProduct( FQIDCref fqid, int coefficient )
{
  SubstancePtr aSubstance( getSuperSystem()->getRootSystem()->
			   getSubstance( fqid ) );
  
  appendProduct( *aSubstance, coefficient );
}

void Reactor::appendCatalyst( FQIDCref fqid,int coefficient)
{
  SubstancePtr aSubstance( getSuperSystem()->getRootSystem()->
			   getSubstance( fqid ) );
  
  appendCatalyst( *aSubstance, coefficient );
}

void Reactor::appendEffector( FQIDCref fqid, int coefficient )
{
  SubstancePtr aSubstance( getSuperSystem()->getRootSystem()->
			   getSubstance( fqid ) );
  
  appendEffector( *aSubstance, coefficient );
}

void Reactor::setInitialActivity( Float activity )
{
  theInitialActivity = activity;
  // FIXME: take delta T from supersystem
  theActivity= activity * 
    getSuperSystem()->getRootSystem()->getStepperLeader().getDeltaT();
}

Reactor::Reactor() 
  :
  theInitialActivity( 0 ),
  theActivityBuffer( 0 ),
  theCondition( Premature ),
  theActivity( 0 )
{
  makeSlots();
}

const String Reactor::getFqpi() const
{
  return PrimitiveTypeStringOf( *this ) + ":" + getFqid();
}

void Reactor::appendSubstrate( SubstanceRef substrate, int coefficient )
{
  ReactantPtr reactant = new Reactant( substrate, coefficient );
  theSubstrateList.insert( theSubstrateList.end(), reactant );
}

void Reactor::appendProduct( SubstanceRef product, int coefficient )
{
  ReactantPtr reactant = new Reactant( product, coefficient );
  theProductList.insert( theProductList.end(), reactant );
}

void Reactor::appendCatalyst( SubstanceRef catalyst, int coefficient )
{
  ReactantPtr reactant = new Reactant( catalyst, coefficient );
  theCatalystList.insert( theCatalystList.end(), reactant );
}

void Reactor::appendEffector( SubstanceRef effector, int coefficient )
{
  ReactantPtr reactant = new Reactant( effector, coefficient );
  theEffectorList.insert( theEffectorList.end(), reactant );
}

Reactor::Condition Reactor::condition( Condition condition )
{
  theCondition = static_cast<Condition>( theCondition | condition );
  if( theCondition  != Good )
    return theGlobalCondition = Bad;
  return Good;
}

void Reactor::warning( StringCref message )
{
//FIXME:   *theMessageWindow << className() << " [" << fqen() << "]";
//FIXME:   *theMessageWindow << ":\n\t" << message << "\n";
}

void Reactor::initialize()
{
  if( getNumberOfSubstrates() > getMaximumNumberOfSubstrates() )
    warning("too many substrates.");
  else if( getNumberOfSubstrates() < getMinimumNumberOfSubstrates() )
    warning("too few substrates.");
  if( getNumberOfProducts() > getMaximumNumberOfProducts() )
    warning("too many products.");
  else if( getNumberOfProducts() < getMinimumNumberOfProducts() )
    warning("too few products.");
  if( getNumberOfCatalysts() > getMaximumNumberOfCatalysts() )
    warning("too many catalysts.");
  else if( getNumberOfCatalysts() < getMinimumNumberOfCatalysts() )
    warning("too few catalysts.");
  if( getNumberOfEffectors() > getMaximumNumberOfEffectors() )
    warning("too many effectors.");
  else if( getNumberOfEffectors() < getMinimumNumberOfEffectors() )
    warning("too few effectors.");
}


Float Reactor::getActivity() 
{
  return theActivity;
}


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
