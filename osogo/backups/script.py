#! /usr/bin/env python

import string
from ecssupport import *

###########################################
#### Simuator Object For Test Mode end ####
###########################################

class simulator :

    def __init__(self) :
        self.theATP={
            'PropertyList': ( 'PropertyList', 'Activity', 'Quantity', 'Concentration','ClassName' ),
            'ClassName' : ('Substance',),
            'Name' : ('ATP Molecule',),
            'Activity' : (100, ),
            'Quantity' : (15, ),
            'Concentration' : (0.0017, )
            }
    
        self.theADP={
            'PropertyList': ( 'PropertyList', 'Activity', 'Quantity', 'Concentration','ClassName' ),
            'ClassName' : ('Substance',),
            'Name' : ('ADP Molecule',),
            'Activity' : (120, ),
            'Quantity' : (30, ),
            'Concentration' : (0.0318318, )
            }
    
        self.theAMP={
            'PropertyList': ( 'PropertyList', 'Activity', 'Quantity', 'Concentration','ClassName' ),
            'ClassName' : ('Substance',),
            'Name' : ('AMP Molecule',),
            'Activity' : (777, ),
            'Quantity' : (40, ),
            'Concentration' : (0.0037, )
            }

        self.theAaa={
            'PropertyList': ( 'PropertyList', 'Activity', 'Quantity', 'Concentration','ClassName' ),
            'ClassName' : ('Substance',),
            'Name' : ('Aaa Molecule',),
            'Activity' : (100, ),
            'Quantity' : (45, ),
            'Concentration' : (0.03103, )
            }

        self.theBbb={
            'PropertyList': ( 'PropertyList', 'Activity', 'Quantity', 'Concentration','ClassName' ),
            'ClassName' : ('Substance',),
            'Name' : ('Bbb Molecule',),
            'Activity' : (38976, ),
            'Quantity' : (18394, ),
            'Concentration' : (0.001083, )
            }

        self.theCcc={
            'PropertyList': ( 'PropertyList', 'Activity', 'Quantity', 'Concentration','ClassName' ),
            'ClassName' : ('Substance',),
            'Name' : ('Ccc Molecule',),
            'Activity' : (938, ),
            'Quantity' : (896, ),
            'Concentration' : (0.082136, )
            }

        self.theDdd={
            'PropertyList': ( 'PropertyList', 'Activity', 'Quantity', 'Concentration','ClassName' ),
            'ClassName' : ('Substance',),
            'Name' : ('Ddd Molecule',),
            'Activity' : (765938, ),
            'Quantity' : (89696, ),
            'Concentration' : (0.0782136, )
            }

        self.theEee={
            'PropertyList': ( 'PropertyList', 'Activity', 'Quantity', 'Concentration','ClassName' ),
            'ClassName' : ('Substance',),
            'Name' : ('Eee Molecule',),
            'Activity' : (9978638, ),
            'Quantity' : (89876, ),
            'Concentration' : (0.09682136, )
            }

        self.theAAA={
            'PropertyList': ( 'PropertyList', 'Activity', 'Km', 'Vmax', 'Substrate', 'Product', 'ClassName' ),
            'Activity': ( 123, ),
            'Km' : ( 1.233, ),
            'Vmax' : ( 349, ),
            'Substrate': ('Substance:/CELL/CYTOPLASM:ATP',
                          'Substance:/CELL/CYTOPLASM:ADP', ),
            'Product': ('Substance:/CELL/CYTOPLASM:AMP', ),
            'ClassName' : 'MichaelisMentenReactor',
            'Name' : ('AAA Reactor',)
            }

        self.theBBB={
            'PropertyList': ( 'PropertyList', 'Activity', 'Km', 'Vmax', 'Substrate', 'Product', 'ClassName' ),
            'Activity': ( 123, ),
            'Km' : ( 1.233, ),
            'Vmax' : ( 349, ),
            'Substrate': ('Substance:/ENVIRONMENT:Ccc',),
            'Product': ('Substance:/ENVIRONMENT:Ddd', 'Substance:/ENVIRONMENT:Eee' ),
            'ClassName' : 'MichaosUniUniReactor',
            'Name' : ('BBB Reactor',)
            }

        self.theCytoplasm={
            'PropertyList': ( 'PropertyList', 'SystemList', 'SubstanceList', 'ReactorList', 'ClassName', 'Activity' ),
            'SystemList' : ( ) ,
            'SubstanceList' : ( 'ATP', 'ADP', 'AMP'),
            'ReactorList' : ( 'AAA', ),
            'ClassName': ( 'System', ),
            'Name' : ('Cytoplasm System',),
            'ATP' : self.theATP,
            'ADP' : self.theADP, 
            'AMP' : self.theAMP,
            'AAA' : self.theAAA
            }

        self.theCell={
            'PropertyList': ( 'PropertyList', 'SystemList', 'SubstanceList', 'ReactorList', 'ClassName', 'Activity' ),
            'SystemList' : ( 'CYTOPLASM', ),
            'SubstanceList' : ( 'Aaa', 'Bbb' ),
            'ReactorList' : ( ),
            'ClassName': ( 'System', ),
            'Name' : ('Cell System',),
            'CYTOPLASM' : self.theCytoplasm,
            'Aaa' : self.theAaa,
            'Bbb' : self.theBbb,
            }
        
        self.theEnvironment={
            'PropertyList': ( 'PropertyList', 'SystemList', 'SubstanceList', 'ReactorList', 'ClassName', 'Activity' ),
            'SystemList' : ( ) ,
            'SubstanceList' : ( 'Ccc', 'Ddd', 'Eee'),
            'ReactorList' : ( 'BBB', ),
            'ClassName': ( 'System', ),
            'Name' : ('Environtment System',),
            'Ccc' : self.theCcc,
            'Ddd' : self.theDdd,
            'Eee' : self.theEee
            }

        self.theRootSystem= {
            'PropertyList': ( 'PropertyList', 'SystemList', 'SubstanceList', 'ReactorList', 'ClassName', 'Activity' ),
            'SystemList' : ( 'CELL', 'ENVIRONMENT' ),
            'SubstanceList' : ( ),
            'ReactorList' : ( ),
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
            
        if fpn[TYPE] == SUBSTANCE :
            aSubstance = aSystem[fpn[ID]]
            return aSubstance[fpn[PROPERTY]]

        elif fpn[TYPE] == REACTOR :
            aReactor = aSystem[fpn[ID]]
            return aReactor[fpn[PROPERTY]]

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
            
        if fpn[TYPE] == SUBSTANCE :
            aSubstance = aSystem[fpn[ID]]
            aSubstance[fpn[PROPERTY]] = arg_list
            print arg_list ,
            print ' is set to ' ,
            print fpn[PROPERTY]

        elif fpn[TYPE] == REACTOR :
            aReactor = aSystem[fpn[ID]]
            aReactor[fpn[PROPERTY]] = arg_list
            print arg_list ,
            print ' is set to ' ,
            print fpn[PROPERTY]

        elif fpn[TYPE] == SYSTEM :
            aSystem[fpn[PROPERTY]] = arg_list
            print arg_list ,
            print ' is set to ' ,
            print fpn[PROPERTY]






