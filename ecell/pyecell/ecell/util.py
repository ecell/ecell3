


def printProperty( s, fullpn ):
    
    value = s.getEntityProperty( fullpn )
    print fullpn, '\t=\t', value

def printAllProperties( s, fullid ):

    plistfullpn = fullid + ':' + 'PropertyList'
    properties = s.getEntityProperty( plistfullpn )
    for property in properties:
        fullpn = fullid + ':' + property[0]
        try:
            printProperty( s, fullpn )
        except:
            print "failed to print %s:%s" % ( fullid, property )

def printStepperProperty( s, id, propertyname ):
    
    value = s.getStepperProperty( id, propertyname )
    print id, ':', propertyname, '\t=\t', value

def printAllStepperProperties( s, id ):

    properties = s.getStepperProperty( id, 'PropertyList' )
    for property in properties:
        try:
            printStepperProperty( s, id, property[0] )
        except:
            print "failed to print %s:%s" % ( id, property )
