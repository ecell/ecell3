#include "MakeSystemCommand.h"

#include "Koyurugi/System.h"
#include "Koyurugi/SystemMaker.h"

MakeSystemCommand::MakeSystemCommand( const string& classname, const string& fqen, const string& name )
  :
  theClassname( classname ),
  theFqen( fqen ),
  theName( name )
{
}


void MakeSystemCommand::doit()
{
  cout<<"this is MakeSystemCommand::doit()."<<endl;

  System* newone = NULL;

  try{
     newone = theRootSystem->systemMaker().make( theClassname );
   }
  catch(...)
    {
      cerr<<"cannot make system"<<endl;
      delete newone;
      throw;
    }

  if( newone == NULL )
    {
      cerr<<"faild to instantiate?"<<endl;
      throw;
    }
  
  FQEN fqen( theFqen );
  
  newone->setEntryname(fqen.entrynameString());
  newone->setName(theName);

  MetaSystem* metasystem;
  try{
    metasystem = dynamic_cast<MetaSystem*>
      (theRootSystem->findSystem(fqen.systemPathString()));
  }
  catch(...){
    cerr<<"error in find system"<<endl;
    delete newone;
    throw;
  }

  if(!metasystem){
    cerr<<"not metasystem"<<endl;
    throw;
  }

  newone->setSupersystem(metasystem);
  metasystem->newSystem(newone);
}

