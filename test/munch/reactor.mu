@{
import string
METHODLIST=[]
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
	print '  registerSlot( getPropertySlotMaker()->createPropertySlot( "%s",*this, Type2Type<Real>(),&%s::set%s, &%s::get%s ));'\
              % (i[1],CLASSNAME,i[1],CLASSNAME,i[1])

def reactantpropertyslotvariabledecls():
    for i in REACTANT_SLOTS:
        print '    PropertySlotPtr %s_%s;' % (i[0],i[1])

def getpropertyslotofreactant():
    for i in REACTANT_SLOTS:
        print '  %s_%s = getPropertySlotOfReactant( "%s", "%s" );' \
            % (i[0],i[1],i[0],i[1])

def allreactantslotsinit():
    if len( ALL_REACTANT_SLOTS ) == 0:
        return

    for i in ALL_REACTANT_SLOTS:
        print 'Reactant_%s.clear()' % i

    print ''
    print '''  for( ReactantMapIterator s( theReactantMap.begin() );
       s != theReactantMap.end() ; ++s )
    {
      SubstancePtr aSubstance( s->second.getSubstance() );
    '''
    for i in ALL_REACTANT_SLOTS:
        print 'Reactant_%s.push_back( aSubstance->getPropertySlot( "%s", this ) );' % (i,i)
    print '}'

def methodDecls():
    for i in METHODLIST:
        print '    void %s();' % i[0]


def methodDefs(name):
    for i in METHODLIST:
	print '''\
void %s::%s()
{
#line %s "%s"
%s
}
'''  % ( CLASSNAME, i[0], i[2][1], i[2][0], i[1] )

def defineMethod( name, content ):
    filename = munch.identify()[0]
    lineno = munch.identify()[1] - len( string.split( content, '\n' ) ) + 1
    METHODLIST.append( ( name, content, ( filename, lineno ) ) )
#    METHODLIST.append( ( name, content) )

def cpplineno():
    print '#line %s "%s"'  % (munch.identify(), munch.identify()[0] )
}@
