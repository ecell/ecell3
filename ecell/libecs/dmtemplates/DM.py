import string

def ECELL_DM_BEGIN( classname, type ):
    filename = identify()[0]
    print "#include \"libecs.hpp\""
    print "#include \"System.hpp\""
    print "#include \"Stepper.hpp\""
    print "#include \"Variable.hpp\""
    print "#include \"VariableProxy.hpp\""
    print "#include \"FluxProcess.hpp\""
    print "#include \"Process.hpp\""
    print "#include \"Util.hpp\""
    print "#include \"PropertyInterface.hpp\""
    print "#include \"PropertySlotMaker.hpp\"\n"
    
    print "#define SIMPLE_SET_GET_METHOD( TYPE, NAME )\\"
    print "SIMPLE_SET_METHOD( TYPE, NAME )\\"
    print "SIMPLE_GET_METHOD( TYPE, NAME )\n"
    print "#define SIMPLE_GET_METHOD( TYPE, NAME )\\"
    print "const TYPE get ## NAME() const\\"
    print "{\\"
    print "return NAME;\\"
    print "}\n"
    
    print "#define SIMPLE_SET_METHOD( TYPE, NAME )\\"
    print "void set ## NAME( TYPE ## Cref value )\\"
    print "{\\"
    print "NAME = value;\\"
    print "}\n"

    print "#define XSTR( S ) STR( S )"
    print "#define STR( S ) #S"
    print "#define ECELL_OBJECT \\"
    print "StringLiteral getClassname() { return XSTR( _ECELL_CLASSNAME ); }\\"
    print "static _ECELL_TYPE* createInstance() { return new _ECELL_CLASSNAME ; }\n"
    print "#define _ECELL_TYPE %s" % type
    print "#define _ECELL_CLASSNAME %s" % classname

    print 'namespace libecs { \n'
    print "#line 1 \"%s\"" % filename
    
def ECELL_DM_END():
    print "extern \"C\""
    print "{"
    print "  _ECELL_TYPE::AllocatorFuncPtr CreateObject ="
    print "  &_ECELL_CLASSNAME::createInstance;"
    print "}"
    print '}'

    print '#undef _ECELL_CLASSNAME\n#undef _ECELL_TYPE'
