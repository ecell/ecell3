#! /usr/bin/python

from ecssupport import *

CREATE = 0
SET = 1

aMainWindow.theCellModelObject = (
( CREATE, 'Variable', ( VARIABLE, '/CELL/CYTOPLASM', 'S2' ), 'substrate' ),
( CREATE, 'Variable', ( VARIABLE, '/CELL/CYTOPLASM', 'P2' ), 'product' ), 
( CREATE, 'Variable', ( VARIABLE, '/CELL/CYTOPLASM', 'E2' ), 'enzyme' ),
( SET, ( VARIABLE, '/CELL/CYTOPLASM', 'S2' , 'Value' ), (300000,) ),
( SET, ( VARIABLE, '/CELL/CYTOPLASM', 'P2' , 'Value' ), (50000,) ),
( SET, ( VARIABLE, '/CELL/CYTOPLASM', 'E2' , 'Value' ), (100,) ),
)




