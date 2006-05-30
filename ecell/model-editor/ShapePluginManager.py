from ShapeDescriptor import *
from ModelEditor import *
from Constants import *
import imp
import os



class ShapePluginManager:
    def __init__(self):
        self.theShapeDescriptors = {}
        theTmpList=[]
        theFileNameList=[]
        theTmpList=os.listdir(SHAPE_PLUGIN_PATH)
        
        for line in theTmpList:
            if line.endswith(".py"):
                newline=line.replace(".py","")
                theFileNameList+=[newline]

        for value in theFileNameList:
            aFp, aPath, self.theDescription = imp.find_module(value ,[SHAPE_PLUGIN_PATH,'.'] )
            module = imp.load_module( value,aFp,aPath, self.theDescription )
            aShapeType = module.SHAPE_PLUGIN_TYPE
            if aShapeType not in self.theShapeDescriptors.keys():
                self.theShapeDescriptors[aShapeType] = {}
            self.theShapeDescriptors[aShapeType][ module.SHAPE_PLUGIN_NAME ] = module
               
                        

    def getShapeList(self,aShapeType):
        return self.theShapeDescriptors[ aShapeType ].keys()


    def createShapePlugin(self,aShapeType,aShapeName,EditorObject,graphUtils, aLabel):
        aShapeModule=self.theShapeDescriptors[aShapeType][aShapeName]
        aShapeSD=apply(aShapeModule.__dict__[os.path.basename(aShapeModule.__name__)],[EditorObject,graphUtils,aLabel])  
        return aShapeSD
        
    def getMinDims( self, aShapeType, aShapeName, graphUtils, aLabel ):
        aShapeModule=self.theShapeDescriptors[aShapeType][aShapeName]
        return apply( aShapeModule.__dict__["estLabelDims"],[graphUtils, aLabel] )
            
        

           
        
    
