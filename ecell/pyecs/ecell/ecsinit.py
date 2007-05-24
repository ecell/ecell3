import os

if os.name != "nt":
    import sys
    import DLFCN
    
    # RTLD_GLOBAL is needed so that rtti across dynamic modules can work
    # RTLD_LAZY   may be needed so that the system can resolve dependency among
    #             dynamic modules after dlopened it
    
    sys.setdlopenflags( DLFCN.RTLD_LAZY | DLFCN.RTLD_GLOBAL )

