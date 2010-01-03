#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2010 Keio University
#       Copyright (C) 2005-2009 The Molecular Sciences Institute
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-Cell System is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
# 
# E-Cell System is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public
# License along with E-Cell System -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
# 
#END_HEADER

import os
import imp

import ecell.ui.model_editor.Config as config
from ecell.ui.model_editor.ShapeDescriptor import *
from ecell.ui.model_editor.ModelEditor import *
from ecell.ui.model_editor.Constants import *

class ShapePluginManager:
    def __init__(self):
        self.theShapeDescriptors = {}
        theTmpList=[]
        theFileNameList=[]
        theTmpList=os.listdir( config.SHAPE_PLUGIN_PATH )
        
        for line in theTmpList:
            if line.endswith(".py"):
                newline=line.replace(".py","")
                theFileNameList+=[newline]

        for value in theFileNameList:
            aFp, aPath, self.theDescription = imp.find_module(value,[ config.SHAPE_PLUGIN_PATH, '.' ] )
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
            
        

           
        
    
