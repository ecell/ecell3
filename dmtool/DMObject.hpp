


/// an allocator function template

template< class Base, class Derived >
Base* ObjectAllocator()
{
  return new Derived;
}



#define DM_INIT( CLASSNAME, TYPE )\
  extern "C"\
  {\
    TYPE::AllocatorFuncPtr CreateObject =\
    &ObjectAllocator<TYPE,CLASSNAME>;\
  } // 


#define DM_OBJECT( CLASSNAME, TYPE )\
 static TYPE* createInstance() { return new CLASSNAME ; }


#define DM_BASECLASS( CLASSNAME )\
public:\
 typedef CLASSNAME * (* AllocatorFuncPtr )()
