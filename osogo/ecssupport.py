#!/usr/bin/env python

import string

SETABLE = 1 << 0   # == 1
GETABLE = 1 << 1   # == 2


class ID:

    def __init__(self, name):
        list = string.split(name, ':')

    def getType(self):
        return self.type

    def getSystemPath(self):
        return self.SystemPath

    def getID(self):
        return self.ID

    def getPropertyName(self):
        return self.PropertyName

    def __getitem__(self, key):
        keyl = string.lower(key)

class FullID( ID ):
    
    def __init__(self, name):
        self.fid = name
        list = string.split(name, ':')
        self.type =  list[0]
        self.SystemPath =  list[1]
        self.ID =  list[2]

    def getFullPropertyName(self, PropertyName = ''):
        self.fpn = self.fid + ':' + PropertyName
        return self.fpn

    def convertToFullPropertyName(self, PropertyName = ''):
        fpn = self.getFullPropertyName(PropertyName)
        return FullPropertyName(fpn)

    def __getitem__(self, key):
        keyl = string.lower(key)
        if keyl == 'type':
            return self.type
        elif keyl == 'systempath':
            return self.SystemPath
        elif keyl == 'id':
            return self.ID
        elif keyl == 'fullpropertyname':
            return self.getFullPropertyName()

    
class FullPropertyName( ID ):

    def __init__(self, name):
        list = string.split(name, ':')
        self.type =  list[0]
        self.SystemPath =  list[1]
        self.ID =  list[2]
        self.PropertyName = list[3]

    def getFullID(self):
        return self.type + ':' + self.SystemPath + ':' + self.ID

    def convertToFullID(self):
        fid = self.getFullID
        return FullID(fpn)

    def __getitem__(self, key):
        keyl = string.lower(key)
        if keyl == 'type':
            return self.type
        elif keyl == 'systempath':
            return self.SystemPath
        elif keyl == 'id':
            return self.ID
        elif keyl == 'propertyname':
            return self.PropertyName
        elif keyl == 'fullid':
            return self.getFullID()


if __name__ == "__main__":
    
    fid = 'Substance:/CELL/CYTOPLASM:ADP'
    i = FullID(fid)
    
    print i['type']
    print i['SystemPath']
    print i['ID']
    print i['fullpropertyname']
    print i.getFullPropertyName('Concentration')

    j = i.convertToFullPropertyName('Concentration')
    print j['PropertyName']
    
    print '------------------------------------------'

    fpn = 'Substance:/CELL/CYTOPLASM:ATP:Quantity'
    f = FullPropertyName(fpn)
    
    print f['type']
    print f['SystemPath']
    print f['ID']
    print f['PropertyName']
    print f['fullID']

    g = j.convertToFullID()
    print g['SystemPath']







