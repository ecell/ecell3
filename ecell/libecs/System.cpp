
char const System_C_rcsid[] = "$Id$";
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




#include "Koyurugi/System.h"
#include "Koyurugi/Reactor.h"
#include "Koyurugi/CellComponents.h"
//FIXME: #include "ecell/MessageWindow.h"
#include "Koyurugi/RootSystem.h"
#include "Koyurugi/Stepper.h"
#include "Koyurugi/StepperMaker.h"
#include "Koyurugi/FQPN.h"

// instantiate primitive lists.
template SubstanceList;
template ReactorList;
template SystemList;
//template GenomicElementList;





/////////////////////// System

void System::makeSlots()
{
  MessageSlot("Stepper",System,*this,&System::setStepper,&System::getStepper);
  MessageSlot("VolumeIndex",System,*this,
	      &System::setVolumeIndex,&System::getVolumeIndex);
}

System::System()
:_volumeIndexName(NULL),_volumeIndex(NULL),_stepper(NULL) 
{
  makeSlots();
}

System::~System()
{
  delete _stepper;
}

const string System::fqpn() const
{
  return Primitive::PrimitiveTypeString(Primitive::SYSTEM) + ":" + fqen();
}

void System::setStepper(const Message& message)
{
  setStepper(message.body(0));
}

const Message System::getStepper(const string& keyword)
{
  return Message(keyword,stepper()->className());
}

void System::setVolumeIndex(const Message& message)
{
  setVolumeIndex(FQEN(message.body()));
}

const Message System::getVolumeIndex(const string& keyword)
{
  if(!volumeIndex())
    return Message(keyword,"");

  return Message(keyword,volumeIndex()->fqen());
}

void System::setStepper(const string& classname)
{
  Stepper* stepper;
  stepper = theRootSystem->stepperMaker().make(classname);
  stepper->setOwner(this);

  _stepper = stepper;
}

Float System::volume() 
{
  return _volumeIndex->activityPerSec();
}

void System::setVolumeIndex(const FQEN& volumeindex)
{
  _volumeIndexName = new FQEN(volumeindex);
}


Primitive System::getPrimitive(const string& entryname,Primitive::Type type)
throw(UnmatchedSystemClass,InvalidPrimitiveType,NotFound)
{
  union {
    SSystem* ssystem;
    RSystem* rsystem;
    MetaSystem* metasystem;
//    Genome* genome;    
  } s;

  Primitive primitive;

  primitive.type = type;

  switch(type)
    {
    case Primitive::SUBSTANCE:
      if(!(s.ssystem = dynamic_cast<SSystem*>(this)))
	throw UnmatchedSystemClass(__PRETTY_FUNCTION__,"[" 
				   + fqen() + "]: this is not a SSystem");
      primitive.substance = s.ssystem->getSubstance(entryname);
      break;
    case Primitive::REACTOR:
      if(!(s.rsystem = dynamic_cast<RSystem*>(this)))
	throw UnmatchedSystemClass(__PRETTY_FUNCTION__,"[" + 
				   fqen() + "]: this is not a RSystem");
      primitive.reactor = s.rsystem->getReactor(entryname);
      break;
    case Primitive::SYSTEM:
      if(!(s.metasystem = dynamic_cast<MetaSystem*>(this)))
	throw UnmatchedSystemClass(__PRETTY_FUNCTION__,"[" + 
				   fqen() + "]: this is not a MetaSystem");
      primitive.system = s.metasystem->getSystem(entryname);
      break;
/*    case Primitive::GENOMICELEMENT:
      if(!(s.genome = dynamic_cast<Genome*>(this)))
	throw UnmatchedSystemClass(__PRETTY_FUNCTION__,"[" + 
				   fqen() + "]: this is not a Genome");
      primitive.genomicElement = s.genome->getGenomicElement(entryname);
      break;*/
    case Primitive::PRIMITIVE_NONE:
    default:
	throw InvalidPrimitiveType(__PRETTY_FUNCTION__,"[" 
				   + fqen() + "]: request type invalid.");
    }

  return primitive;
}

int System::sizeOfPrimitiveList(Primitive::Type type)
{
  union {
    SSystem* ssystem;
    RSystem* rsystem;
    MetaSystem* metasystem;
//    Genome* genome;
  } s;
  switch(type)
    {
    case Primitive::SUBSTANCE:
      if(!(s.ssystem = dynamic_cast<SSystem*>(this)))
	return 0;
      return s.ssystem->sizeOfSubstanceList();
    case Primitive::REACTOR:
      if(!(s.rsystem = dynamic_cast<RSystem*>(this)))
	return 0;
      return s.rsystem->sizeOfReactorList();
    case Primitive::SYSTEM:
      if(!(s.metasystem = dynamic_cast<MetaSystem*>(this)))
	return 0;
      return s.metasystem->sizeOfSystemList();
/*    case Primitive::GENOMICELEMENT:
      if(!(s.genome = dynamic_cast<Genome*>(this)))
	return 0;
      return s.genome->sizeOfGenomicElementList();*/
    case Primitive::PRIMITIVE_NONE:
    default:
	throw InvalidPrimitiveType(__PRETTY_FUNCTION__,"[" 
				   + fqen() + "]: request type invalid");
    }

  throw UnexpectedError(__PRETTY_FUNCTION__);
}

void System::forAllPrimitives(Primitive::Type type,PrimitiveCallback cb,
			      void* clientData)
{
  assert(cb);

  union {
    SSystem* ssystem;
    RSystem* rsystem;
    MetaSystem* metasystem;
//    Genome* genome;
  } s;

  SubstanceListIterator si = NULL;
  ReactorListIterator ri = NULL;
  SystemListIterator yi = NULL;
//  GenomicElementListIterator gi = NULL;

  Primitive* primitive;

  switch(type)
    {
    case Primitive::SUBSTANCE:
      if(!(s.ssystem = dynamic_cast<SSystem*>(this)))
	throw UnmatchedSystemClass(__PRETTY_FUNCTION__,"[" 
				   + fqen() + "]: this is not a SSystem");
      for(si = s.ssystem->firstSubstance(); 
	  si != s.ssystem->lastSubstance(); si++)
	{
	  primitive = new Primitive(si->second);
	  cb(primitive,clientData);
	  delete primitive;
	}
      return;
    case Primitive::REACTOR:
      if(!(s.rsystem = dynamic_cast<RSystem*>(this)))
	throw UnmatchedSystemClass(__PRETTY_FUNCTION__,"[" 
				   + fqen() + "]: this is not a RSystem");
      for(ri = s.rsystem->firstReactor(); ri != s.rsystem->lastReactor(); ri++)
	{
	  primitive = new Primitive(ri->second);
	  cb(primitive,clientData);
	  delete primitive;
	}
      return;
    case Primitive::SYSTEM:
      if(!(s.metasystem = dynamic_cast<MetaSystem*>(this)))
	throw UnmatchedSystemClass(__PRETTY_FUNCTION__,"[" 
				   + fqen() + "]: this is not a MetaSystem");
      for(yi = s.metasystem->firstSystem(); 
	  yi != s.metasystem->lastSystem(); yi++)
	{
	  primitive = new Primitive(yi->second);
	  cb(primitive,clientData);
	  delete primitive;
	}
      return;
/*    case Primitive::GENOMICELEMENT:
      if(!(s.genome = dynamic_cast<Genome*>(this)))
	throw UnmatchedSystemClass(__PRETTY_FUNCTION__,"[" 
				   + fqen() + "]: this is not a Genome");
      for(gi = s.genome->firstGenomicElement(); 
	  gi != s.genome->lastGenomicElement(); gi++)
	{
	  primitive = new Primitive(*gi);
	  cb(primitive,clientData);
	  delete primitive;
	}
      return;*/
    case Primitive::PRIMITIVE_NONE:
    default:
	throw InvalidPrimitiveType(__PRETTY_FUNCTION__,"[" 
				   + fqen() + "]: request type invalid");
    }
} 

void System::initialize()
{
  assert(_stepper);

  try{
    if(_volumeIndexName != NULL)
      {
	FQPN fqpn(Primitive::REACTOR,*_volumeIndexName);
	Primitive p(theRootSystem->getPrimitive(fqpn));
	_volumeIndex = p.reactor;
	//FIXME: *theMessageWindow << fqen() << ": volume index is [" 
	//FIXME: 	  << _volumeIndex->fqen() << "].\n";

      }
    else
      {
	//FIXME: *theMessageWindow << fqen() << ": no volume index is specified.\n"; 
      }
  }
  catch(NotFound)
    {
      //FIXME: *theMessageWindow << fqen() << ": volume index [" 
	//FIXME: << _volumeIndexName->fqenString() << "] not found.\n";
    }

  delete _volumeIndexName;
}




/////////////////////// RSystem

RSystem::RSystem()
{
//  _reactorList.clear();
  _firstRegularReactor = _reactorList.begin();
}

bool RSystem::newReactor(Reactor *reactor)
{
  assert(reactor);

  if(containsReactor(reactor->entryname()))
    {
      //FIXME: *theMessageWindow << "multiple definition of reactor [" 
	//FIXME: << reactor->entryname() << "] on [" << entryname() << 
	  //FIXME: "], later one discarded.\n";
      return false;
    }

  _reactorList[reactor->entryname()] = reactor;
  return true;
}

void RSystem::initialize()
{
  for(ReactorListIterator i = firstReactor() ; i != lastReactor() ; i++)
    i->second->initialize();

  _firstRegularReactor = find_if(firstReactor(),lastReactor(),
				 isRegularReactorItem());
}


void RSystem::react()
{
#ifdef DEBUG_STEPPER
  cerr << fqen() <<": react() " << endl;
#endif /* DEBUG_STEPPER */
  for(ReactorListIterator i = firstRegularReactor() ; i != lastReactor() ; i++)
    i->second->react();
}

void RSystem::transit()
{
  for(ReactorListIterator i = firstRegularReactor() ; i != lastReactor() ; i++)
    i->second->transit();
}

void RSystem::postern()
{
#ifdef DEBUG_STEPPER
  cerr << fqen() <<": postern() " << endl;
#endif /* DEBUG_STEPPER */
  for(ReactorListIterator i = firstReactor() ; 
      i != firstRegularReactor() ; i++)
    i->second->react();

  // update activity of posterior reactors by buffered values 
  for(ReactorListIterator i = firstReactor() ; 
      i != firstRegularReactor() ; i++)
    i->second->transit();
}



Reactor* RSystem::getReactor(const string& e_name) throw(NotFound)
{
  ReactorListIterator it = getReactorIterator(e_name);
  if(it == lastReactor())
    throw NotFound(__PRETTY_FUNCTION__, "[" + fqen() + 
		   "]: Reactor [" + e_name + "] not found in this System.");
  return it->second;
}



///////////////////// SSystem

SSystem::SSystem()
{
//  _substanceList.clear();
}

bool SSystem::newSubstance(Substance* newone)
{
  if(containsSubstance(newone->entryname()))
    {
//FIXME:       *theMessageWindow << "multiple definition of substance [" 
//FIXME: 	<< newone->entryname() << "] on [" << entryname() << 
//FIXME: 	  "], name and quantity overwrote.\n";
      _substanceList[newone->entryname()]->setName(newone->name());
      Message m("Quantity",newone->quantity());
      _substanceList[newone->entryname()]->set(m);
//      _substanceList[newone->entryname()]->setQuantity(newone->quantity());
      delete newone;
      return false;
    }
  _substanceList[newone->entryname()] = newone;
  newone->setSupersystem(this);
  return true;
}

void SSystem::initialize()
{
  for(SubstanceListIterator i = _substanceList.begin() ; 
      i != _substanceList.end() ; i++)
    i->second->initialize();
}

void SSystem::clear()
{
  for(SubstanceListIterator i = _substanceList.begin() ; 
      i != _substanceList.end() ; i++)
    i->second->clear();
}

void SSystem::turn()
{
  for(SubstanceListIterator i = _substanceList.begin() ; 
      i != _substanceList.end() ; i++)
    i->second->turn();
}

void SSystem::transit()
{
  for(SubstanceListIterator i = _substanceList.begin() ; 
      i != _substanceList.end() ; i++)
    i->second->transit();
}

void SSystem::fixSubstance(const string& entry)
{
  Substance* s;
  if( (s = getSubstance(entry)) == NULL)
    {
//FIXME:       *theMessageWindow << "fix request to undefined substance [" 
//FIXME: 	<< entry << "] on [" << entryname() << "]. ignoring.\n";
      return; 
    }
  s->fix();
}

Substance* SSystem::getSubstance(const string& e_name) throw(NotFound)
{
  SubstanceListIterator it = getSubstanceIterator(e_name);
  if(it == lastSubstance())
    throw NotFound(__PRETTY_FUNCTION__, "[" + fqen() + 
		   "]: Substance [" + e_name + "] not found in this System.");
  return it->second;
}


////////////////////// MetaSystem

MetaSystem::MetaSystem()
{
//  _subsystemList.clear();
}


bool MetaSystem::newSystem(System* system)
{
  if(containsSystem(system->entryname()))
    {
//FIXME:       *theMessageWindow << "multiple definition of system [" 
//FIXME: 	<< system->entryname() << "] on [" << entryname() << 
//FIXME: 	  "], later one discarded.\n";
      delete system;
      return false;
    }

  _subsystemList[system->entryname()] = system;
  return true;
}

void MetaSystem::initialize()
{
  for(SystemListIterator s = firstSystem();s != lastSystem(); s++)
    {
      s->second->System::initialize();
      s->second->initialize();
    }
}

System* MetaSystem::findSystem(const SystemPath& systempath) 
throw(NotFound)
{
  System* s = getSystem(systempath.first());
  SystemPath next = systempath.next();

  if(next.systemPathString() == "") // a leaf
    return s;
  
  // an node, not a leaf
  MetaSystem* ms = dynamic_cast<MetaSystem*>(s);
  if(!ms)
    throw NotFound(__PRETTY_FUNCTION__,
		   "attempt to get a subsystem of non-MetaSystem [" +
		   next.systemPathString() + "] in [" + fqen() + "].");

  return ms->findSystem(next);
}

System* MetaSystem::getSystem(const string& e_name) throw(NotFound)
{
  SystemListIterator it = getSystemIterator(e_name);
  if(it == lastSystem())
    throw NotFound(__PRETTY_FUNCTION__, "[" + fqen() + 
		   "]: System [" + e_name + "] not found in this System.");
  return it->second;
}

