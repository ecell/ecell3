#! /usr/bin/python

from ecssupport import *

CREATE = 0
SET = 1

aMainWindow.theCellModelObject = (
( CREATE, 'System', ( SYSTEM, '/', 'CELL' ), 'cell'), 
( CREATE, 'System', ( SYSTEM, '/CELL', 'CYTOPLASM' ), 'The cytoplasm' ), 
( CREATE, 'System', ( SYSTEM, '/CELL', 'MEMBRANE' ), 'membrane' ), 
( CREATE, 'System', ( SYSTEM, '/', 'ENVIRONMENT' ), 'environment' ), 
( SET, ( SYSTEM, '/CELL', 'CYTOPLASM', 'Volume' ), (10e-17,) ), 
( CREATE, 'Substance', ( SUBSTANCE, '/CELL/CYTOPLASM', 'S' ), 'substrate' ),
( CREATE, 'Substance', ( SUBSTANCE, '/CELL/CYTOPLASM', 'P' ), 'product' ), 
( CREATE, 'Substance', ( SUBSTANCE, '/CELL/CYTOPLASM', 'E' ), 'enzyme' ),
( CREATE, 'Substance', ( SUBSTANCE, '/ENVIRONMENT', 'GLU' ), 'Glucose', ), 
( CREATE, 'Substance', ( SUBSTANCE, '/ENVIRONMENT', 'PYR' ), 'Pyruvate' ), 
( CREATE, 'Substance', ( SUBSTANCE, '/ENVIRONMENT', 'LCT' ), 'Lactate' ), 
( CREATE, 'Substance', ( SUBSTANCE, '/CELL/MEMBRANE', 'C1' ), 'Channel 1' ),
( CREATE, 'Substance', ( SUBSTANCE, '/CELL/MEMBRANE', 'C2' ), 'Channel 2' ),
( CREATE, 'Substance', ( SUBSTANCE, '/CELL/MEMBRANE', 'C3' ), 'Channel 3' ),
( CREATE, 'MichaelisUniUniReactor', ( REACTOR, '/CELL/CYTOPLASM', 'R' ), 'Reaction 1' ),
( SET, ( SUBSTANCE, '/CELL/CYTOPLASM', 'S' , 'Quantity' ), (100000,) ),
( SET, ( SUBSTANCE, '/CELL/CYTOPLASM', 'P' , 'Quantity' ), (60000,) ),
( SET, ( SUBSTANCE, '/CELL/CYTOPLASM', 'E' , 'Quantity' ), (300,) ),
( SET, ( SUBSTANCE, '/ENVIRONMENT', 'GLU', 'Quantity' ), (1453,) ), 
( SET, ( SUBSTANCE, '/ENVIRONMENT', 'PYR', 'Quantity' ), (2430,) ), 
( SET, ( SUBSTANCE, '/ENVIRONMENT', 'LCT', 'Quantity' ), (2134,) ), 
( SET, ( SUBSTANCE, '/CELL/MEMBRANE', 'C1', 'Quantity' ), (124,) ), 
( SET, ( SUBSTANCE, '/CELL/MEMBRANE', 'C2', 'Quantity' ), (321,) ), 
( SET, ( SUBSTANCE, '/CELL/MEMBRANE', 'C3', 'Quantity' ), (242,) ), 
( SET, ( REACTOR, '/CELL/CYTOPLASM', 'R', 'AppendSubstrate' ), ('Substance:/CELL/CYTOPLASM:S',1) ), 
( SET, ( REACTOR, '/CELL/CYTOPLASM', 'R', 'AppendProduct' ), ('Substance:/CELL/CYTOPLASM:P',1) ), 
( SET, ( REACTOR, '/CELL/CYTOPLASM', 'R', 'AppendCatalyst' ), ('Substance:/CELL/CYTOPLASM:E',1) ), 
( SET, ( REACTOR, '/CELL/CYTOPLASM', 'R', 'KmS' ), (.01,) ), 
( SET, ( REACTOR, '/CELL/CYTOPLASM', 'R', 'KcF' ), (1,) ), 

)




