#! /usr/bin/python

from ecssupport import *

CREATE = 0
SET = 1

aCellModelObject = (
( CREATE, 'System', ( SYSTEM, '/', 'CELL' ), 'cell'), 
( CREATE, 'System', ( SYSTEM, '/CELL', 'CYTOPLASM' ), 'The cytoplasm' ), 
( CREATE, 'System', ( SYSTEM, '/CELL', 'MEMBRANE' ), 'membrane' ), 
( CREATE, 'System', ( SYSTEM, '/', 'ENVIRONMENT' ), 'environment' ), 
( SET, ( SYSTEM, '/CELL', 'CYTOPLASM', 'Volume' ), (10e-17,) ), 
( CREATE, 'Variable', ( VARIABLE, '/CELL/CYTOPLASM', 'S' ), 'substrate' ),
( CREATE, 'Variable', ( VARIABLE, '/CELL/CYTOPLASM', 'P' ), 'product' ), 
( CREATE, 'Variable', ( VARIABLE, '/CELL/CYTOPLASM', 'E' ), 'enzyme' ),
( CREATE, 'Variable', ( VARIABLE, '/ENVIRONMENT', 'GLU' ), 'Glucose', ), 
( CREATE, 'Variable', ( VARIABLE, '/ENVIRONMENT', 'PYR' ), 'Pyruvate' ), 
( CREATE, 'Variable', ( VARIABLE, '/ENVIRONMENT', 'LCT' ), 'Lactate' ), 
( CREATE, 'Variable', ( VARIABLE, '/CELL/MEMBRANE', 'C1' ), 'Channel 1' ),
( CREATE, 'Variable', ( VARIABLE, '/CELL/MEMBRANE', 'C2' ), 'Channel 2' ),
( CREATE, 'Variable', ( VARIABLE, '/CELL/MEMBRANE', 'C3' ), 'Channel 3' ),
( CREATE, 'MichaelisUniUniProcess', ( PROCESS, '/CELL/CYTOPLASM', 'R' ), 'Reaction 1' ),
( SET, ( VARIABLE, '/CELL/CYTOPLASM', 'S' , 'Value' ), (100000,) ),
( SET, ( VARIABLE, '/CELL/CYTOPLASM', 'P' , 'Value' ), (60000,) ),
( SET, ( VARIABLE, '/CELL/CYTOPLASM', 'E' , 'Value' ), (300,) ),
( SET, ( VARIABLE, '/ENVIRONMENT', 'GLU', 'Value' ), (1453,) ), 
( SET, ( VARIABLE, '/ENVIRONMENT', 'PYR', 'Value' ), (2430,) ), 
( SET, ( VARIABLE, '/ENVIRONMENT', 'LCT', 'Value' ), (2134,) ), 
( SET, ( VARIABLE, '/CELL/MEMBRANE', 'C1', 'Value' ), (124,) ), 
( SET, ( VARIABLE, '/CELL/MEMBRANE', 'C2', 'Value' ), (321,) ), 
( SET, ( VARIABLE, '/CELL/MEMBRANE', 'C3', 'Value' ), (242,) ), 
( SET, ( PROCESS, '/CELL/CYTOPLASM', 'R', 'AppendSubstrate' ), ('Variable:/CELL/CYTOPLASM:S',1) ), 
( SET, ( PROCESS, '/CELL/CYTOPLASM', 'R', 'AppendProduct' ), ('Variable:/CELL/CYTOPLASM:P',1) ), 
( SET, ( PROCESS, '/CELL/CYTOPLASM', 'R', 'AppendCatalyst' ), ('Variable:/CELL/CYTOPLASM:E',1) ), 
( SET, ( PROCESS, '/CELL/CYTOPLASM', 'R', 'KmS' ), (.01,) ), 
( SET, ( PROCESS, '/CELL/CYTOPLASM', 'R', 'KcF' ), (1,) ), 

)




