


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


# deprecated, should not be used
def isNumber( s, aFullPN ):

    aValue = self.getProperty( s, aFullPN )
    if type( aValue[0] ) is types.IntType:
        return 1
    elif type( aValue[0] ) is types.FloatType:
        return 1
    else:
        return 0
