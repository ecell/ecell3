

#define DM_INIT( TYPE, CLASSNAME )\
  extern "C"\
  {\
    TYPE::AllocatorFuncPtr CreateObject =\
    &CLASSNAME::createInstance;\
  }\
}


#define DM_OBJECT( TYPE, CLASSNAME )\
 static TYPE* createInstance() { return new CLASSNAME ; }


#define DM_BASECLASS( CLASSNAME )\
 typedef ProcessPtr (* AllocatorFuncPtr )()
