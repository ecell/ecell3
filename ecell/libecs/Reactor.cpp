
char const Reactor_C_rcsid[] = "$Id$";
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
// 		This file is part of Serizawa (E-CELL Core System)
//
//	       written by Kouichi Takahashi  <shafi@sfc.keio.ac.jp>
//
//                              E-CELL Project,
//                          Lab. for Bioinformatics,  
//                             Keio University.
//
//             (see http://www.e-cell.org for details about E-CELL)
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// Serizawa is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// Serizawa is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with Serizawa -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER




//FIXME: #include "ecell/MessageWindow.h"
#include "Koyurugi/Reactant.h"
#include "Koyurugi/RootSystem.h"
#include "Koyurugi/Stepper.h"
#include "Koyurugi/FQPN.h"

#include "Koyurugi/Reactor.h"

Reactor::Condition Reactor::_globalCondition;// = Reactor::Condition::Good;
const char* Reactor::LIGAND_STRING_TABLE[] = {"substrate","product","catalyst"
					"effector",NULL};

void Reactor::makeSlots()
{
  // No get methods for them. They should be get by 
  // usual methods like substrate(int).
  MessageSlot("Substrate",Reactor,*this,&Reactor::setSubstrate,NULL);
  MessageSlot("Product",Reactor,*this,&Reactor::setProduct,NULL);
  MessageSlot("Catalyst",Reactor,*this,&Reactor::setCatalyst,NULL);
  MessageSlot("Effector",Reactor,*this,&Reactor::setEffector,NULL);
  MessageSlot("InitialActivity",Reactor,*this,&Reactor::setInitialActivity,
	      &Reactor::getInitialActivity);
}

void Reactor::setSubstrate(const Message& message)
{
  setSubstrate(message.body(0),asInt(message.body(1)));
}

void Reactor::setProduct(const Message& message)
{
  setProduct(message.body(0),asInt(message.body(1)));
}

void Reactor::setCatalyst(const Message& message)
{
  setCatalyst(message.body(0),asInt(message.body(1)));
}

void Reactor::setEffector(const Message& message)
{
  setEffector(message.body(0),asInt(message.body(1)));
}

void Reactor::setInitialActivity(const Message& message)
{
  setInitialActivity(asFloat(message.body(0)));
}

const Message Reactor::getInitialActivity(const string& keyword)
{
  return Message(keyword,_initialActivity);
}

void Reactor::setSubstrate(const FQEN& fqen,int coefficient)
{
  FQPN fqpn(Primitive::SUBSTANCE,fqen);
  Primitive p = theRootSystem->getPrimitive(fqpn);
  
  addSubstrate(*(p.substance),coefficient);
}

void Reactor::setProduct(const FQEN& fqen,int coefficient)
{
  FQPN fqpn(Primitive::SUBSTANCE,fqen);
  Primitive p = theRootSystem->getPrimitive(fqpn);
  
  addProduct(*(p.substance),coefficient);
}

void Reactor::setCatalyst(const FQEN& fqen,int coefficient)
{
  FQPN fqpn(Primitive::SUBSTANCE,fqen);
  Primitive p = theRootSystem->getPrimitive(fqpn);
  
  addCatalyst(*(p.substance),coefficient);
}

void Reactor::setEffector(const FQEN& fqen,int coefficient)
{
  FQPN fqpn(Primitive::SUBSTANCE,fqen);
  Primitive p = theRootSystem->getPrimitive(fqpn);
  
  addEffector(*(p.substance),coefficient);
}

void Reactor::setInitialActivity(Float activity)
{
  _initialActivity = activity;
//  _activity= activity * supersystem()->stepper()->deltaT();
  _activity= activity * theRootSystem->stepperLeader().deltaT();
}

Reactor::Reactor() 
:_initialActivity(0),_activityBuffer(0),_condition(Premature),_activity(0)
{
  makeSlots();
}

const string Reactor::fqpn() const
{
  return Primitive::PrimitiveTypeString(Primitive::REACTOR) + ":" + fqen();
}

void Reactor::addSubstrate(Substance& substrate,int coefficient)
{
  Reactant* reactant = new Reactant(substrate,coefficient);
  _substrateList.insert(_substrateList.end(),reactant);
}

void Reactor::addProduct(Substance& product,int coefficient)
{
  Reactant* reactant = new Reactant(product,coefficient);
  _productList.insert(_productList.end(),reactant);
}

void Reactor::addCatalyst(Substance& catalyst,int coefficient)
{
  Reactant* reactant = new Reactant(catalyst,coefficient);
  _catalystList.insert(_catalystList.end(),reactant);
}

void Reactor::addEffector(Substance& effector,int coefficient)
{
  Reactant* reactant = new Reactant(effector,coefficient);
  _effectorList.insert(_effectorList.end(),reactant);
}

Reactor::Condition Reactor::condition(Condition condition)
{
  _condition = static_cast<Condition>(_condition | condition);
  if(_condition  != Good)
    return _globalCondition = Bad;
  return Good;
}


void Reactor::warning(const string& message)
{
//FIXME:   *theMessageWindow << className() << " [" << fqen() << "]";
//FIXME:   *theMessageWindow << ":\n\t" << message << "\n";
}

void Reactor::initialize()
{
  if(numSubstrate() > maxSubstrate())
    warning("too many substrates.");
  else if(numSubstrate() < minSubstrate())
    warning("too few substrates.");
  if(numProduct() > maxProduct())
    warning("too many products.");
  else if(numProduct() < minProduct())
    warning("too few products.");
  if(numCatalyst() > maxCatalyst())
    warning("too many catalysts.");
  else if(numCatalyst() < minCatalyst())
    warning("too few catalysts.");
  if(numEffector() > maxEffector())
    warning("too many effectors.");
  else if(numEffector() < minEffector())
    warning("too few effectors.");
}


Float Reactor::activity() 
{
  return _activity;
}
