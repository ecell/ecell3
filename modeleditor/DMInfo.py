

import ecell._ecs
from Config import *
import os
import os.path
from Constants import *
from Utils import *
import sys
import string

class DMInfo:
    # FIRST INITIATE HARDCODED CLASSES SUCH AS VARIABLE, SYSTEM, COMPARTMENT SYSTMEM FROM HERE
    def __init__(self ):
        self.theList={}
        self.theClass=None
        self.theProcessClassList = None
        self.theStepperClassList = None

    # SECOND DO THIS
    def getClassList( self, aType ):
        if aType == ME_SYSTEM_TYPE:
            return [DM_SYSTEM_CLASS, DM_SYSTEM_CLASS_OLD ]
        if aType == ME_VARIABLE_TYPE:
            return [DM_VARIABLE_CLASS ]
        if aType == ME_PROCESS_TYPE and self.theProcessClassList != None:
            return self.theProcessClassList
        if aType == ME_STEPPER_TYPE and self.theStepperClassList != None:
            return self.theStepperClassList

        aList = []
        # get from current directory for other type of classes
        curdir = '.'
        filelist = os.listdir( curdir )
        # get from module directory
        filelist.extend( os.listdir( DM_PATH ) )

        for aFile in filelist:
            if aFile.endswith( aType + '.desc' ):
                aList.append( aFile.replace( '.desc' , '') )
        if aType == ME_PROCESS_TYPE :
            self.theProcessClassList = aList
        if aType == ME_STEPPER_TYPE:
            self.theStepperClassList = aList
        return aList


    #THIRD
    def getClassInfoList( self, aClass ):
        aFullID = self.__getFullID( aClass )
        if aFullID == None:
            return []
        return [DM_DESCRIPTION, DM_PROPERTYLIST, DM_ACCEPTNEWPROPERTY]



    #FIVE
    def getClassInfo( self, aClass, anInfo ):
        # verify dictionary
        aKey=aClass
        if aClass==DM_SYSTEM_CLASS or aClass==DM_SYSTEM_CLASS_OLD:
            aKey='System'
        if not self.theList.has_key(aKey):
            self.__readFromFile(aClass)
        
        aFullID = self.__getFullID( aClass )
        if aFullID ==None:
            raise Exception("%s class doesnt have %s info"%(aClass, anInfo) )
        
        # read from dictionary
        aClassInfo=self.theList[aClass]
        for aKey in aClassInfo:
            if aKey==anInfo:
                if anInfo == DM_PROPERTYLIST:
                    return self.__getPropertyNameList(aClassInfo[aKey],'name')
                else:
                    return aClassInfo[aKey]


    def getClassPropertyInfo( self, aClass, aProperty, anInfo ):
        aFullID = self.__getFullID( aClass )
        self.theClass=aClass
        if aFullID ==None:
            raise Exception("%s class doesnt have %s info"%(aClass, anInfo) )
        
        if anInfo == DM_PROPERTY_DEFAULTVALUE:
            aType = self.getClassPropertyInfo( aClass, aProperty, DM_PROPERTY_TYPE )
            if aType == DM_PROPERTY_STRING:
                return ''
            elif aType == DM_PROPERTY_NESTEDLIST:
                return []
            else:
                return self.__getPropertyInfo(aProperty,anInfo)

        elif anInfo == DM_PROPERTY_SETTABLE_FLAG:
            flagAttr=self.__getPropertyInfo(aProperty,anInfo)
            return eval(flagAttr[ME_SETTABLE_FLAG])

        elif anInfo == DM_PROPERTY_GETTABLE_FLAG:
            flagAttr=self.__getPropertyInfo(aProperty,anInfo)
            return eval(flagAttr[ME_GETTABLE_FLAG])

        elif anInfo == DM_PROPERTY_LOADABLE_FLAG:
            flagAttr=self.__getPropertyInfo(aProperty,anInfo)
            return eval(flagAttr[ME_LOADABLE_FLAG])

        elif anInfo == DM_PROPERTY_SAVEABLE_FLAG:
            flagAttr=self.__getPropertyInfo(aProperty,anInfo)
            return eval(flagAttr[ME_SAVEABLE_FLAG])

        elif anInfo == DM_PROPERTY_DELETEABLE_FLAG:
            return False

        elif anInfo == DM_PROPERTY_TYPE:
            return self.__getPropertyInfo(aProperty,anInfo)
            

    #-------------------------------------------------------------------------------------
    #
    #-------------------------------------------------------------------------------------


    def __getFullID( self, aClass):
        # get type
        if aClass.endswith('System'):
            aType = ME_SYSTEM_TYPE
        elif aClass.endswith(DM_VARIABLE_CLASS):
            aType = ME_VARIABLE_TYPE
        elif aClass.endswith('Process'):
            aType = ME_PROCESS_TYPE
        elif aClass.endswith('Stepper'):
            aType = ME_STEPPER_TYPE
        else:
            print  "%s classname is ambigouos"%aClass
            return None

        if aClass not in self.getClassList(aType):
            raise Exception("Unknown class: %s"%aClass)

        aFullID = aType + ':/:' + aClass
        return aFullID

    
    def __getPropertyInfo(self,aProperty,anInfo):
        aProplist=self.theList[self.theClass][DM_PROPERTYLIST]
        aName=None
        end=0
        start=0
        until=0
        aType=None
        aDefault=None
        aFlag=[]
        aFlagstr=None
        for target in aProplist:
            end=target.find(',')
            if end!=-1:
                aName=(target[0:end]).strip()
                if aName==aProperty:
                    if anInfo==DM_PROPERTY_TYPE:
                        start=end+1
                        until=  target.find(',',start)
                        if until!=-1:
                            aType=(target[start:until]).strip()
                            break
                    elif anInfo==DM_PROPERTY_DEFAULTVALUE:
                        start=end+1
                        until=  target.find(',',start)
                        start=until+1
                        until=target.find(',',start)
                        if until!=-1:
                            aDefault=(target[start:until]).strip()
                            break
                    else:
                        start=target.rfind(',')
                        until=len(target)
                        aFlagstr=(target[start+1:until]).strip()
                        
                            
        if anInfo==DM_PROPERTY_TYPE:
            return aType
        elif anInfo==DM_PROPERTY_DEFAULTVALUE:
            return aDefault
        else:
            for i in range(len(aFlagstr)):
                aFlag.append(aFlagstr[i])
            return aFlag

    def __getPropertyNameList(self,aList,anAttr):
        attrList=[]
        start=0
        end=0
        aName=None
        if anAttr=='name':
            for target in aList:
                end=target.find(',')
                if end!=-1:
                    aName=(target[0:end]).strip()
                    attrList.append(aName)
        return attrList 
    
    def __checkFileFormat(self,content):
        isDesc=False
        isAccept=False
        isProp=False
        isPropEnd=False
        for line in content:
            if line.startswith(DM_DESCRIPTION + '='):
                isDesc=True
            if line.startswith(DM_ACCEPTNEWPROPERTY+ '='):
                isAccept=True
            if line.startswith(DM_PROPERTYLIST):
                isProp=True
            if line.startswith('END'):
                isPropEnd=True  
        if not isDesc:
            return False
        if not isAccept:
            return False
        if not isProp:
            return False
        if not isPropEnd:
            return False
        return True

    def __readFromFile(self,aClass):
        noExclude=0
        aProplist=[]
        aDescription=None
        acceptNew=False
        isAdd=False

        curdir = '.'
        filelist = os.listdir( curdir )
        aFileName=''
        

        # set file name 
        if aClass.endswith('System'):
            aFileName=DM_PATH +'/' + aClass[-6:]+'.desc'
        else:   
            targetfile=aClass+'.desc'
            for filename in filelist:
                if filename==targetfile:
                    aFileName=os.path.join(curdir,filename)
                    break
            if aFileName=='':
                aFileName=DM_PATH +'/' + targetfile


        
        aFileDesc=open(aFileName,'r')
        content = aFileDesc.readlines()
        
        #check file format
        if not self.__checkFileFormat(content):
            raise Exception ("Please check the format of file %s.!"%aFileName)

        for record in content:
            if record.startswith(DM_DESCRIPTION):
                noExclude=len(DM_DESCRIPTION)+1
                aDescription=(record[noExclude:]).strip()

            elif record.startswith(DM_ACCEPTNEWPROPERTY):
                noExclude=len(DM_ACCEPTNEWPROPERTY)+1
                aValue=(record[noExclude:]).strip()
                if aValue=='True':
                    acceptNew=True
                else:
                    acceptNew=False

            elif record.startswith(DM_PROPERTYLIST):
                isAdd=True
            if isAdd:
                aProplist.append(record)    
        aFileDesc.close()
        
        # remove '\n' from Property
        del aProplist[0]
        del aProplist[len(aProplist)-1]
        aTemplist=[]
        for aProp in aProplist:
            if aProp.endswith('\n'):
                aTemplist.append(aProp[:-1])
        aTemplist.sort()
        aProplist=aTemplist

        # addToDictionary
        aDict=dict([(DM_DESCRIPTION,aDescription),(DM_ACCEPTNEWPROPERTY,acceptNew),(DM_PROPERTYLIST,aProplist)])
        (aDict.keys()).sort()
        self.theList[aClass]=aDict
        (self.theList.keys()).sort()



def DMTypeCheck( aValue, aType ):

    if aType == DM_PROPERTY_STRING:
        if type(aValue) == type([]):
            aValue = aValue[0]
        return str( aValue )
    elif aType == DM_PROPERTY_MULTILINE:
        if type(aValue) == type([]):
            aValue = '/n'.join(aValue)

        return str( aValue )
    elif aType == DM_PROPERTY_NESTEDLIST:
        if type(aValue) in ( type( () ), type( [] ) ):
            return aValue
        else:
            return None
    elif aType == DM_PROPERTY_INTEGER:
        if type(aValue) == type([]):
            aValue = aValue[0]
        try:
            aValue = int( aValue )
        except:
            return None
        
        return aValue
    elif aType == DM_PROPERTY_FLOAT:
        if type(aValue) == type([]):
            aValue = aValue[0]
        try:
            aValue = float( aValue )
        except:
            return None
        return aValue
    else:
        return None

