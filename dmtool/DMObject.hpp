


/// an allocator function template

template< class Base, class Derived >
Base* ObjectAllocator()
{
  return new Derived;
}



#define DM_INIT( CLASSNAME, TYPE )\
  extern "C"\
  {\
    ECELL_API TYPE::AllocatorFuncPtr CreateObject =\
    &ObjectAllocator<TYPE,CLASSNAME>;\
    const char* __DM_CLASSNAME = #CLASSNAME;\
    const char* __DM_TYPE = #TYPE;\
    ECELL_API const void *(*GetClassInfo)() = &CLASSNAME::getClassInfoPtr;\
  } // 


#define DM_OBJECT( CLASSNAME, TYPE )\
 static TYPE* createInstance() { return new CLASSNAME ; }


#define DM_BASECLASS( CLASSNAME )\
public:\
 typedef CLASSNAME * (* AllocatorFuncPtr )()
