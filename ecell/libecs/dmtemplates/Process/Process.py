import string
METHOD_LIST={}
INCLUDE_FILES=[]
CONNECTION_SLOTS=[]
ALL_CONNECTION_SLOTS=[]
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
	print '  registerSlot( getPropertySlotMaker()->createPropertySlot( "%s",*this, Type2Type<%s>(),&%s::set%s, &%s::get%s ));'\
              % (i[1],i[0],CLASSNAME,i[1],CLASSNAME,i[1])

def connectionpropertyslotvariabledecls():
    for i in CONNECTION_SLOTS:
        print '    PropertySlotPtr %s_%s;' % (i[0],i[1])
    for i in ALL_CONNECTION_SLOTS:
        print '    std::vector<PropertySlotPtr> Connection_%s;' % i

def getpropertyslotofconnection():
    for i in CONNECTION_SLOTS:
        print '  %s_%s = getPropertySlotOfConnection( "%s", "%s" );' \
            % (i[0],i[1],i[0],i[1])

def allconnectionslotsinit():
    if len( ALL_CONNECTION_SLOTS ) == 0:
        return

    for i in ALL_CONNECTION_SLOTS:
        print 'Connection_%s.clear();' % i

    print ''
    print '''  for( ConnectionMapIterator s( theConnectionMap.begin() );
       s != theConnectionMap.end() ; ++s )
    {
      VariablePtr aVariable( s->second.getVariable() );
    '''
    for i in ALL_CONNECTION_SLOTS:
        print 'Connection_%s.push_back( aVariable->getPropertySlot( "%s", this ) );' % (i,i)
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

