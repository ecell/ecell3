from Constants import *

def copyValue ( aValue ):
    """
    in: anytype aValue
    return copy of aValue (coverts tuples to list)
    """
    if type (aValue) == type (()) or type( aValue ) == type([]):
        returnValue = []        
        for anElement in aValue:
            returnValue.append( copyValue( anElement ) )             
             
        return returnValue
    else:
              
        return aValue

def getFullIDType( aFullID ):
    aList = aFullID.split(':')
    return aList[0]


def getClassTypeFromName( aClassName ):
    for aType in [ ME_SYSTEM_TYPE, ME_VARIABLE_TYPE, ME_PROCESS_TYPE, ME_STEPPER_TYPE]:
        if aClassName.endswith( aType):
            return aType
    return None


def getParentSystemOfFullID( aFullID ):
    return convertSysPathToSysID( aFullID.split(':')[1] )



def convertSysIDToSysPath( aSystemFullID ):
    if aSystemFullID == 'System::/':
        return  '/'

    aList = aSystemFullID.split(':')
    if aList[1] == '/':
        aList[1] = ''
    return aList[1] + '/' + aList[2]


def convertSysPathToSysID( aSystemPath ):
    if aSystemPath == '/':
        return  'System::/'
        
    aPos = aSystemPath.rfind('/')
    newSysID = ['System', aSystemPath[0:aPos], aSystemPath[aPos+1:len(aSystemPath) ] ]
    if newSysID[1] == '':
        newSysID[1] = '/'
    return ":".join(newSysID)


def getAbsoluteReference( aProcessFullID, aVariableRef ):
    if isAbsoluteReference( aVariableRef ):
        return aVariableRef
    aVariable = aVariableRef.split(':')
    if aVariable[1][0] == '/':
        # absolute ref
        absolutePath = aVariable[1]
    elif aVariable[1][0] == '.':
        aProcess = aProcessFullID.split(':')[1]
        aProcessPath = aProcess.split('/')
        while True:
            if len(aProcessPath) == 0:
                break
            if aProcessPath[0] == '':
                aProcessPath.__delitem__(0)
            else:
                break
        aVariablePath = aVariable[1].split('/')
        absolutePath = ''
        while aVariablePath != []:
            pathString =  aVariablePath.pop()
            if pathString == '.':
                break
            elif pathString == '..':
                if len(aProcessPath) == 0:
                    raise Exception("BROKEN REFERENCE")
                aProcessPath.pop()
            else:
                absolutePath =  pathString + '/' + absolutePath
        oldPath = '/' + '/'.join(aProcessPath)
        absolutePath = absolutePath.rstrip('/')
        if oldPath != '/' and absolutePath != '':
            oldPath +='/'
        absolutePath =  oldPath + absolutePath

    else:
        raise Exception("INVALID REFERENCE")

    return aVariable[0] + ':' + absolutePath + ':' + aVariable[2]


def getRelativeReference( aProcessFullID, aVariableFullID ):
    # get pathes from each
    processPath = aProcessFullID.split(':')[1]
    processPath = processPath.replace('/','',1)
    processTupple = processPath.split('/')

    variablePath = aVariableFullID.split(':')[1]
    variablePath = variablePath.replace('/','',1)
    variableTupple = variablePath.split('/')

    # delete matching path elements
    while len(processTupple)>0 and len(variableTupple)>0:
        if processTupple[0] == variableTupple[0]:
            processTupple.__delitem__(0)
            variableTupple.__delitem__(0)
        else:
            break
    relPath = ''
    # create up directries
    if len(processTupple) == 0:
        processTupple = ['']
    if len(processTupple) == 1 and  processTupple[0]=='':
        relPath = './'
    else:
        for aPath in processTupple:
            relPath += '../'

    
    # create down directoriwes
    for aPath in variableTupple:
        relPath += aPath + '/'
    relPath = relPath.strip('/')
    varref=aVariableFullID.split(':')
    varref[1] = relPath
    return ':'.join( varref )


def getMinPath( path1, path2 ):
    list1 = path1.split('/')
    list2 = path2.split('/')
    list3 = []
    while len(list1)>0 and len(list2)>0:
        if list1[0] == list2[0]:
            list3.append( list1[0] )
            list1.__delitem__(0)
            list2.__delitem__(0)
        else:
            break
    if len(list3) == 1:
        return '/'
    return '/'.join(list3)


def isAbsoluteReference( aVariableRef ):
    aList = aVariableRef.split(':')
    return aList[1][0] == '/'


def isRelativeReference( aVariableRef ):
    aList = aVariableRef.split(':')
    return aList[1][0] == '.'


def createFullPN( aFullID, aPropertyName ):
    return aFullID + ':' + aPropertyName


def getFullID( aFullPN ):
    aList = aFullPN.split(':')
    return aList[0] + ':' + aList[1] + ':' +aList[2]


def getPropertyName( aFullPN ):
    aList = aFullPN.split(':')
    return aList[3]


def createFullIDList( aType, aParent, IDList):
    parentPath = convertSysIDToSysPath( aParent)
    return_list = []
    for anID in IDList:
        return_list.append( ':'.join( [aType, parentPath, anID] ) )
    return return_list

def getUniqueVarrefName ( aVarrefList, aVarrefName = None ):
    # get existing varrefnamelist
    nameList = []
    for aVarref in aVarrefList:
        nameList.append( aVarref[MS_VARREF_NAME] )

    # get initial varrefname
    if aVarrefName == None:
        aVarrefName = '___'
    elif aVarrefName not in nameList:
        return aVarrefName
    counter = 0 
    # try other values
    incVarrefName = aVarrefName + zfill( str( counter ) )
    while incVarrefName in nameList:
        incVarrefName = aVarrefName + zfill( str( counter ) )
    return incVarrefName
    
def createFullIDFromVarref( aProcessFullID, aVarref ):
    #aVarref: containing all 3 components
    aVarrefFullID = aVarref[ MS_VARREF_FULLID ]
    aVarrefFullID = getAbsoluteReference( aProcessFullID,  aVarrefFullID )
    aVarrefTuple = aVarrefFullID.split( ':' )
    aVarrefTuple[0] = 'Variable'
    return ':'.join( aVarrefTuple )


