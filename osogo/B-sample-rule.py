#! /usr/bin/python

from ecssupport import *

CREATE = 0
SET = 1

aMainWindow.theCellModelObject = (
( CREATE, 'Substance', ( SUBSTANCE, '/CELL/CYTOPLASM', 'S2' ), 'substrate' ),
( CREATE, 'Substance', ( SUBSTANCE, '/CELL/CYTOPLASM', 'P2' ), 'product' ), 
( CREATE, 'Substance', ( SUBSTANCE, '/CELL/CYTOPLASM', 'E2' ), 'enzyme' ),
( SET, ( SUBSTANCE, '/CELL/CYTOPLASM', 'S2' , 'Quantity' ), (300000,) ),
( SET, ( SUBSTANCE, '/CELL/CYTOPLASM', 'P2' , 'Quantity' ), (50000,) ),
( SET, ( SUBSTANCE, '/CELL/CYTOPLASM', 'E2' , 'Quantity' ), (100,) ),
)




