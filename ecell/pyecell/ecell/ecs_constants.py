
#
# constants for ecell
#


# boolean constants
TRUE  = 1
FALSE = 0

# FullPN field numbers
TYPE       = 0
SYSTEMPATH = 1
ID         = 2
PROPERTY   = 3

# Entity type numbers
ENTITY     = 1
VARIABLE  = 2
PROCESS    = 3
SYSTEM     = 4

ENTITYTYPE_STRING_LIST =\
( 'NONE', 'Entity', 'Variable', 'Process', 'System' )


ENTITYTYPE_DICT =\
{
    'Entity'   : ENTITY,
    'Variable': VARIABLE,
    'Process'  : PROCESS,
    'System'   : SYSTEM
}    

# File extension

MODEL_FILE_EXTENSION = 'eml' 
SCRIPT_FILE_EXTENSION = 'ess' 


