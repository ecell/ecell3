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




#include "assert.h"
#include "CellComponents.h"
//FIXME: #include "ecell/MessageWindow.h"
#include "RootSystem.h"
#include "SystemMaker.h"


//// register all cell components to SystemMaker

void SystemMaker::makeClassList()
{
  NewSystemModule(Environment);
  NewSystemModule(Cell);
  NewSystemModule(Cytoplasm);
  NewSystemModule(Membrane);
}


/////////////////////// Environment

Environment::Environment()
{
//
}

void Environment::initialize()
{
  SSystem::initialize();
  RSystem::initialize();
}

void Environment::clear()
{
  SSystem::clear();             // clear velocity of all substances by zero
}

void Environment::react()
{
  RSystem::react();           // change concentration by velocity
}

void Environment::transit()
{
  RSystem::transit();
  SSystem::transit();           // change concentration by velocity
}

void Environment::postern()
{
  RSystem::postern();
}



////////////////////////// Monolithic

Monolithic::Monolithic()
{
}

void Monolithic::initialize()
{
  SSystem::initialize();
  RSystem::initialize();
}

void Monolithic::clear()
{
  SSystem::clear();             // clear velocity of all substances by zero
}

void Monolithic::react()
{
  RSystem::react();           // call react() method of each reactor
}
void Monolithic::transit()
{
  RSystem::transit();
  SSystem::transit();           // change concentration by velocity
}

void Monolithic::postern()
{
  RSystem::postern();
}

////////////////////////// Cytoplasm

Cytoplasm::Cytoplasm()
{
}

void Cytoplasm::initialize()
{
  SSystem::initialize(); 
  RSystem::initialize();
  MetaSystem::initialize();
}

void Cytoplasm::clear()
{
  SSystem::clear();             // clear velocity of all substances by zero
}

void Cytoplasm::react()
{
  RSystem::react();           // call react() method of each reactor
}
void Cytoplasm::transit()
{
  RSystem::transit();
  SSystem::transit();           // change concentration by velocity
}

void Cytoplasm::postern()
{
  RSystem::postern();
}

////////////////////////// Membrane

void Membrane::makeSlots()
{
  MessageSlot("Inside",Membrane,*this,&Membrane::setInside,&Membrane::getInside);
  MessageSlot("Outside",Membrane,*this,&Membrane::setOutside,&Membrane::getOutside);
}

void Membrane::setInside(const Message& message)
{
  setInside(message.body());
}

void Membrane::setOutside(const Message& message)
{
  setOutside(message.body());
}

void Membrane::setInside(StringCref systemname)
{
  FQIN fqen(systemname);

  //FIXME: handle exception
  MetaSystem* s = dynamic_cast<MetaSystem*>
    (theRootSystem->findSystem(fqen.systemPathString()));
  if(!s)
    {
      _inside = NULL;
      return;
    }
  System* sys = s->getSystem(fqen.entrynameString());

  _inside = sys;

}

void Membrane::setOutside(StringCref systemname)
{
  FQIN fqen(systemname);

  //FIXME: handle exception
  MetaSystem* s = dynamic_cast<MetaSystem*>
    (theRootSystem->findSystem(fqen.systemPathString()));
  if(!s)
    {
      _inside = NULL;
      return;
    }
  System* sys = s->getSystem(fqen.entrynameString());

  _inside = sys;
}

const Message Membrane::getInside(StringCref keyword)
{
  if(!_inside)
    return Message(keyword,"");

  return Message(keyword,inside()->fqen());
}

const Message Membrane::getOutside(StringCref keyword)
{
  if(!_outside)
    return Message(keyword,"");
  return Message(keyword,outside()->fqen());
}

Membrane::Membrane()
{
  makeSlots();
}

void Membrane::initialize()
{
  SSystem::initialize();
  RSystem::initialize();
}

void Membrane::clear()
{
  SSystem::clear();             // clear velocity of all substances by zero
}

void Membrane::react()
{
  RSystem::react();           // call react() method of each reactor
}

void Membrane::transit()
{
  RSystem::transit();
  SSystem::transit();           // change concentration by velocity
}

void Membrane::postern()
{
  RSystem::postern();
}


//////////////////////// Cell


Cell::Cell()
{
}

void Cell::initialize()
{
  SSystem::initialize();
  RSystem::initialize();
  MetaSystem::initialize();
}

void Cell::clear()
{
  SSystem::clear();             // clear velocitys of all substances by zero
}

void Cell::react()
{
  RSystem::react();           // call react() method of each reactor
}                                

void Cell::transit()
{
  RSystem::transit();
  SSystem::transit();           // change concentration by velocity
}

void Cell::postern()
{
  RSystem::postern();
}


#if 0

//////////////////////// Chromosome

Chromosome::Chromosome()
{

}

void Chromosome::initialize()
{
  SSystem::initialize(); 
  RSystem::initialize(); 
}

void Chromosome::clear()
{
  SSystem::clear();             // clear velocitys of all substances by zero
}

void Chromosome::react()
{
  RSystem::react();           // call react() method of each reactor
}                                

void Chromosome::transit()
{
  RSystem::transit();
  SSystem::transit();           // change concentration by velocity
}

//////////////////////// GXSystem

GXSystem::GXSystem()
{
  _activityIndex = NULL;
}

Float GXSystem::activity()
{
  if(_activityIndex == NULL) 
    return -1;
  return _activityIndex->activity();
}

void GXSystem::initialize()
{
  RSystem::initialize();           
}                                

void GXSystem::clear()
{
//  SSystem::clear();            
}

void GXSystem::react()
{
  RSystem::react();           
}                                

void GXSystem::transit()
{
  RSystem::transit();
//  SSystem::transit();         
}





/////////////////////// GenomicElement

GenomicElement::GenomicElement()
{

}

////////////////////// Gene

Gene::Gene()
{
}


#endif /* 0 */
