#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2007 Keio University
#       Copyright (C) 2005-2007 The Molecular Sciences Institute
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

import string
from ecssupport import *

###########################################
#### Simuator Object For Test Mode end ####
###########################################

class simulator :

    def __init__(self) :
        self.theATP={
            'PropertyList': ( 'PropertyList', 'Activity', 'Value', 'Concentration','ClassName' ),
            'ClassName' : ('Variable',),
            'Name' : ('ATP Molecule',),
            'Activity' : (100, ),
            'Value' : (15, ),
            'Concentration' : (0.0017, )
            }
    
        self.theADP={
            'PropertyList': ( 'PropertyList', 'Activity', 'Value', 'Concentration','ClassName' ),
            'ClassName' : ('Variable',),
            'Name' : ('ADP Molecule',),
            'Activity' : (120, ),
            'Value' : (30, ),
            'Concentration' : (0.0318318, )
            }
    
        self.theAMP={
            'PropertyList': ( 'PropertyList', 'Activity', 'Value', 'Concentration','ClassName' ),
            'ClassName' : ('Variable',),
            'Name' : ('AMP Molecule',),
            'Activity' : (777, ),
            'Value' : (40, ),
            'Concentration' : (0.0037, )
            }

        self.theAaa={
            'PropertyList': ( 'PropertyList', 'Activity', 'Value', 'Concentration','ClassName' ),
            'ClassName' : ('Variable',),
            'Name' : ('Aaa Molecule',),
            'Activity' : (100, ),
            'Value' : (45, ),
            'Concentration' : (0.03103, )
            }

        self.theBbb={
            'PropertyList': ( 'PropertyList', 'Activity', 'Value', 'Concentration','ClassName' ),
            'ClassName' : ('Variable',),
            'Name' : ('Bbb Molecule',),
            'Activity' : (38976, ),
            'Value' : (18394, ),
            'Concentration' : (0.001083, )
            }

        self.theCcc={
            'PropertyList': ( 'PropertyList', 'Activity', 'Value', 'Concentration','ClassName' ),
            'ClassName' : ('Variable',),
            'Name' : ('Ccc Molecule',),
            'Activity' : (938, ),
            'Value' : (896, ),
            'Concentration' : (0.082136, )
            }

        self.theDdd={
            'PropertyList': ( 'PropertyList', 'Activity', 'Value', 'Concentration','ClassName' ),
            'ClassName' : ('Variable',),
            'Name' : ('Ddd Molecule',),
            'Activity' : (765938, ),
            'Value' : (89696, ),
            'Concentration' : (0.0782136, )
            }

        self.theEee={
            'PropertyList': ( 'PropertyList', 'Activity', 'Value', 'Concentration','ClassName' ),
            'ClassName' : ('Variable',),
            'Name' : ('Eee Molecule',),
            'Activity' : (9978638, ),
            'Value' : (89876, ),
            'Concentration' : (0.09682136, )
            }

        self.theAAA={
            'PropertyList': ( 'PropertyList', 'Activity', 'Km', 'Vmax', 'Substrate', 'Product', 'ClassName' ),
            'Activity': ( 123, ),
            'Km' : ( 1.233, ),
            'Vmax' : ( 349, ),
            'Substrate': ('Variable:/CELL/CYTOPLASM:ATP',
                          'Variable:/CELL/CYTOPLASM:ADP', ),
            'Product': ('Variable:/CELL/CYTOPLASM:AMP', ),
            'ClassName' : 'MichaelisMentenProcess',
            'Name' : ('AAA Process',)
            }

        self.theBBB={
            'PropertyList': ( 'PropertyList', 'Activity', 'Km', 'Vmax', 'Substrate', 'Product', 'ClassName' ),
            'Activity': ( 123, ),
            'Km' : ( 1.233, ),
            'Vmax' : ( 349, ),
            'Substrate': ('Variable:/ENVIRONMENT:Ccc',),
            'Product': ('Variable:/ENVIRONMENT:Ddd', 'Variable:/ENVIRONMENT:Eee' ),
            'ClassName' : 'MichaosUniUniProcess',
            'Name' : ('BBB Process',)
            }

        self.theCytoplasm={
            'PropertyList': ( 'PropertyList', 'SystemList', 'VariableList', 'ProcessList', 'ClassName', 'Activity' ),
            'SystemList' : ( ) ,
            'VariableList' : ( 'ATP', 'ADP', 'AMP'),
            'ProcessList' : ( 'AAA', ),
            'ClassName': ( 'System', ),
            'Name' : ('Cytoplasm System',),
            'ATP' : self.theATP,
            'ADP' : self.theADP, 
            'AMP' : self.theAMP,
            'AAA' : self.theAAA
            }

        self.theCell={
            'PropertyList': ( 'PropertyList', 'SystemList', 'VariableList', 'ProcessList', 'ClassName', 'Activity' ),
            'SystemList' : ( 'CYTOPLASM', ),
            'VariableList' : ( 'Aaa', 'Bbb' ),
            'ProcessList' : ( ),
            'ClassName': ( 'System', ),
            'Name' : ('Cell System',),
            'CYTOPLASM' : self.theCytoplasm,
            'Aaa' : self.theAaa,
            'Bbb' : self.theBbb,
            }
        
        self.theEnvironment={
            'PropertyList': ( 'PropertyList', 'SystemList', 'VariableList', 'ProcessList', 'ClassName', 'Activity' ),
            'SystemList' : ( ) ,
            'VariableList' : ( 'Ccc', 'Ddd', 'Eee'),
            'ProcessList' : ( 'BBB', ),
            'ClassName': ( 'System', ),
            'Name' : ('Environtment System',),
            'Ccc' : self.theCcc,
            'Ddd' : self.theDdd,
            'Eee' : self.theEee
            }

        self.theRootSystem= {
            'PropertyList': ( 'PropertyList', 'SystemList', 'VariableList', 'ProcessList', 'ClassName', 'Activity' ),
            'SystemList' : ( 'CELL', 'ENVIRONMENT' ),
            'VariableList' : ( ),
            'ProcessList' : ( ),
            'ClassName': ( 'System', ),
            'Name' : ('Root System',),
            'Activity': ( 1234, ),
            'CELL' : self.theCell,
            'ENVIRONMENT' : self.theEnvironment
            }


    def getEntityProperty( self, fpn ):
        aSystemList = string.split(fpn[SYSTEMPATH] , '/')
        aLength = len( aSystemList )
        if fpn[SYSTEMPATH] == '/':
            aSystem = self.theRootSystem
        else :
            aSystem = self.theRootSystem
            for x in aSystemList[1:] :
                aSystem = aSystem[ x ]
            
        if fpn[TYPE] == VARIABLE :
            aVariable = aSystem[fpn[ID]]
            return aVariable[fpn[PROPERTY]]

        elif fpn[TYPE] == PROCESS :
            aProcess = aSystem[fpn[ID]]
            return aProcess[fpn[PROPERTY]]

        elif fpn[TYPE] == SYSTEM :
            if fpn[ID] == '/' and fpn[SYSTEMPATH] == '/':
                aTargetSystem = aSystem
            else:
                aTargetSystem = aSystem[fpn[ID]]
            return aTargetSystem[fpn[PROPERTY]]

    
    def setEntityProperty( self, fpn, arg_list ):
        aSystemList = string.split(fpn[SYSTEMPATH] , '/')
        aLength = len( aSystemList )
        if aLength == 1 :
            aSystem = self.theRootSystem
        else :
            aSystem = self.theRootSystem
            for x in aSystemList[1:] :
                aSystem = aSystem[ x ]
            
        if fpn[TYPE] == VARIABLE :
            aVariable = aSystem[fpn[ID]]
            aVariable[fpn[PROPERTY]] = arg_list
            print arg_list ,
            print ' is set to ' ,
            print fpn[PROPERTY]

        elif fpn[TYPE] == PROCESS :
            aProcess = aSystem[fpn[ID]]
            aProcess[fpn[PROPERTY]] = arg_list
            print arg_list ,
            print ' is set to ' ,
            print fpn[PROPERTY]

        elif fpn[TYPE] == SYSTEM :
            aSystem[fpn[PROPERTY]] = arg_list
            print arg_list ,
            print ' is set to ' ,
            print fpn[PROPERTY]






