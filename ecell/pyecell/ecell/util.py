


def printProperty( s, fullpn ):
    
    value = s.getProperty( fullpn )
    print fullpn, '\t=\t', value

def printAllProperties( s, fullid ):

    plistfullpn = fullid + ':' + 'PropertyList'
    properties = s.getProperty( plistfullpn )
    for property in properties:
        fullpn = fullid + ':' + property
        try:
            printProperty( s, fullpn )
        except:
            print "failed to print %s:%s" % ( fullid, property )

