import string
METHOD_LIST={}
INCLUDE_FILES=[]
VARIABLE_SLOTS=[]
ALL_VARIABLE_SLOTS=[]
PUBLIC_AUX = ''
PROTECTED_AUX = ''
PRIVATE_AUX= ''


def fileincludes():
    for i in INCLUDE_FILES:
        print '#include %s' % i

def propertyvariabledecls():
    for i in PROPERTIES:
	print '    %s %s;' % (i[0], i[1])

def propertymethods():
    for i in PROPERTIES:
        print '    void set%s( %sCref value ) { %s = value; }' % (i[1],i[0],i[1])
        print '    const %s get%s() const { return %s; }' % (i[0],i[1],i[1])

def propertyvariableinit():
    for i in PROPERTIES:
        print '  %s = %s;' % (i[1],i[2])

def createpropertyslots():
    for i in PROPERTIES:
	print '  DEFINE_PROPERTYSLOT( %s, %s, &%s::set%s, &%s::get%s );'\
              % (i[0],i[1],CLASSNAME,i[1],CLASSNAME,i[1])

def variablepropertyslotvariabledecls():
    for i in VARIABLE_SLOTS:
        print '    PropertySlotPtr %s_%s;' % (i[0],i[1])
    for i in ALL_VARIABLE_SLOTS:
        print '    std::vector<PropertySlotPtr> Variable_%s;' % i

def getpropertyslotofvariable():
    for i in VARIABLE_SLOTS:
        print '  %s_%s = getPropertySlotOfVariable( "%s", "%s" );' \
            % (i[0],i[1],i[0],i[1])

def allvariableslotsinit():
    if len( ALL_VARIABLE_SLOTS ) == 0:
        return

    for i in ALL_VARIABLE_SLOTS:
        print 'Variable_%s.clear();' % i

    print ''
    print '''  for( VariableReferenceMapIterator s( theVariableReferenceMap.begin() );
       s != theVariableReferenceMap.end() ; ++s )
    {
      VariablePtr aVariable( s->second.getVariable() );
    '''
    for i in ALL_VARIABLE_SLOTS:
        print 'VariableReference_%s.push_back( aVariable->getPropertySlot( "%s", this ) );' % (i,i)
    print '}'

#def methodDecls():
#    for i in METHOD_LIST.keys():
#        print '    void %s();' % i


def methodDefs(name):
    if not METHOD_LIST.has_key( name ):
    	return
    i = METHOD_LIST[name]
    print '''\
#line %s "%s"
%s
'''  % ( i[1][1], i[1][0], i[0] )

def defineMethod( name, content ):
    filename = identify()[0]
#    lineno = identify()[1] # - len( string.split( content, '\n' ) ) + 1

#   give the context, not __FILE__ and __LINE__
    contextname = "%s::%s() in %s" % ( CLASSNAME, name, filename )

    METHOD_LIST[name] = ( content, ( contextname, 1 ) )

def cpplineno():
    print '#line %s "%s"'  % (identify(), identify()[0] )

