#!/usr/bin/env python

import string

class FullPropertyName:

    def __init__(self, fpn):
        list = string.split(fpn, ':')
        self.type =  list[0]
        self.SystemPath =  list[1]
        self.ID =  list[2]
        self.PropertyName = list[3]

    def getType(self):
        return self.type

    def getSystemPath(self):
        return self.SystemPath

    def getID(self):
        return self.ID

    def getPropertyName(self):
        return self.PropertyName

    def getFullID(self):
        return self.type + ':' + self.SystemPath + ':' + self.ID

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
    aFQPP1 = 'Substance:/CELL/CYTOPLASM:ATP:Quantity'

    f = FullPropertyName(aFQPP1)
    
    print f['type']
    print f['SystemPath']
    print f['ID']
    print f['PropertyName']
    print f['fullID']
