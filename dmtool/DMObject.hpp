


/// an allocator function template

template< class Base, class Derived >
Base* ObjectAllocator()
{
  return new Derived;
}



#define DM_INIT( TYPE, CLASSNAME )\
  extern "C"\
  {\
    TYPE::AllocatorFuncPtr CreateObject =\
    &ObjectAllocator<TYPE,CLASSNAME>;\
  } // 


#define DM_OBJECT( TYPE, CLASSNAME )\
 static TYPE* createInstance() { return new CLASSNAME ; }


#define DM_BASECLASS( CLASSNAME )\
 typedef CLASSNAME * (* AllocatorFuncPtr )()
