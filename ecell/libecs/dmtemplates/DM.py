import string

def ECELL_DM_BEGIN( type, classname ):
    filename, line = identify()
    print '''
    #define __ECELL_TYPE %s
    #define __ECELL_CLASSNAME %s
    #define LIBECS_DM_OBJECT( __ECELL_TYPE, __ECELL_CLASSNAME )
    namespace libecs { \n
    #line %s \"%s\"''' % ( type, classname, line, filename )
    
def ECELL_DM_END():
    print '''
    extern \"C\"
    {
      _ECELL_TYPE::AllocatorFuncPtr CreateObject =
        &_ECELL_CLASSNAME::createInstance;
      }
    }
    #undef __ECELL_CLASSNAME
    #undef __ECELL_TYPE
    '''
