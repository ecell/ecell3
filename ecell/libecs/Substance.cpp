
char const Substance_C_rcsid[] = "$Id$";
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




#include "ecscore/Substance.h"
#include "ecscore/Integrators.h"
#include "ecscore/System.h"
#include "ecscore/RootSystem.h"
#include "util/Util.h"
#include "ecscore/Accumulators.h"
#include "ecscore/AccumulatorMaker.h"
#include "ecell/MessageWindow.h"


const string Substance::SYSTEM_DEFAULT_ACCUMULATOR_NAME = "ReserveAccumulator";
string Substance::USER_DEFAULT_ACCUMULATOR_NAME 
= Substance::SYSTEM_DEFAULT_ACCUMULATOR_NAME;

void Substance::makeSlots()
{
  MessageSlot("Quantity",Substance,*this,&Substance::setQuantity,
	      &Substance::getQuantity);
  MessageSlot("Accumulator",Substance,*this,&Substance::setAccumulator,
	      &Substance::getAccumulator);
}

void Substance::setQuantity(const Message& message)
{
  Float f = asFloat(message.body());

  if(_accumulator)
    loadQuantity(f);
  else
    {
      mySetQuantity(f);
    }
}


void Substance::setAccumulator(const Message& message)
{
  setAccumulator(message.body(0));
}

const Message Substance::getQuantity(const string& keyword)
{
  return Message(keyword,quantity());
}

const Message Substance::getAccumulator(const string& keyword)
{
  if(_accumulator)
    return Message(keyword,_accumulator->className());
  else
    return Message(keyword,"");
}

Substance::Substance()
: _accumulator(NULL),_integrator(NULL),_quantity(0),  
  _fraction(0),  _velocity(0), _bias(0),
  _fixed(false) ,_concentration(-1)
{
  makeSlots();
} 

Substance::~Substance()
{
  delete _integrator;
  delete _accumulator;
}

void Substance::setAccumulator(const string& classname)
{
  try {
    Accumulator* a;
    a = theRootSystem->accumulatorMaker().make(classname);
    setAccumulator(a);
    if(classname != userDefaultAccumulatorName())
      *theMessageWindow << "[" << fqpn() 
	<< "]: accumulator is changed to: " << classname << ".\n";
  }
  catch(Exception& e)
    {
      *theMessageWindow << "[" << fqpn() << "]:\n" << e.message();
      if(_accumulator)   // _accumulator is already set
	{
	  *theMessageWindow << "[" << fqpn() << 
	    "]: falling back to :" << _accumulator->className() << ".\n";
	}
    }
}

void Substance::setAccumulator(Accumulator* a)
{
  if(_accumulator)
    delete _accumulator;
  _accumulator = a;
  _accumulator->setOwner(this);
  _accumulator->update();
}

const string Substance::fqpn() const
{
  return Primitive::PrimitiveTypeString(Primitive::SUBSTANCE) + ":" + fqen();
}


void Substance::initialize()
{
  if(!_accumulator)
    setAccumulator(USER_DEFAULT_ACCUMULATOR_NAME);
  if(!_accumulator)  // if the user default is invalid fall back to the
    {                // system default.
      *theMessageWindow << "Substance: " 
	<< "falling back to the system default accumulator: " 
	  << SYSTEM_DEFAULT_ACCUMULATOR_NAME  << ".\n";
      setUserDefaultAccumulatorName(SYSTEM_DEFAULT_ACCUMULATOR_NAME);
      setAccumulator(USER_DEFAULT_ACCUMULATOR_NAME);
    }

}

Float Substance::saveQuantity()
{
  return _accumulator->save();
}

void Substance::loadQuantity(Float q)
{
  mySetQuantity(q);
  _accumulator->update();
}

Float Substance::activity()
{
  return velocity();
}

Concentration Substance::calculateConcentration()
{
  return _concentration = _quantity / (Float)(supersystem()->volume()*N_A); 
}


bool Substance::haveConcentration()
{
  return (supersystem()->volumeIndex()) ? true : false;
}

void Substance::transit()
{ 
  _concentration = -1;

  if(_fixed) 
    return;

  _integrator->transit();

  _accumulator->doit();
  
   if(_quantity < 0) 
     {
       _quantity = 0;
//       throw LTZ();
     }
}

void Substance::clear()
{ 
  _quantity += _bias; 
  _bias = 0;
  _velocity = 0; 
  _integrator->clear();
}

void Substance::turn()
{
  _integrator->turn();
}

Float Substance::velocity(Float v)
{
  _velocity += v;
  return _velocity;
}


