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

#
# constants for ecell
#
from ecell._ecs import EntityType

# Entity type numbers
ENTITY     = EntityType.ENTITY
VARIABLE   = EntityType.VARIABLE
PROCESS    = EntityType.PROCESS
SYSTEM     = EntityType.SYSTEM

ENTITYTYPE_LIST = (
    EntityType.NONE,
    EntityType.ENTITY,
    EntityType.VARIABLE,
    EntityType.PROCESS,
    EntityType.SYSTEM,
    )

ENTITYTYPE_STRING_LIST = tuple([
    EntityType.getString( val ) for val in ENTITYTYPE_LIST ])

ENTITYTYPE_DICT = dict( zip( ENTITYTYPE_STRING_LIST, ENTITYTYPE_LIST ) )

# File extension

MODEL_FILE_EXTENSION = 'eml' 
SCRIPT_FILE_EXTENSION = 'ess' 
ECD_EXTENSION = 'ecd'

# FullPN field numbers
TYPE       = 0
SYSTEMPATH = 1
ID         = 2
PROPERTY   = 3

CLASSINFO_TYPE = 0
CLASSINFO_SETTABLE = 1
CLASSINFO_GETTABLE = 2
CLASSINFO_LOADABLE = 3
CLASSINFO_SAVEABLE = 4

DM_TYPE_STEPPER  = 'Stepper'
DM_TYPE_VARIABLE = 'Variable'
DM_TYPE_PROCESS  = 'Process'
DM_TYPE_SYSTEM   = 'System'


DM_TO_ENTITY_TYPE_MAP = {
    DM_TYPE_VARIABLE: VARIABLE,
    DM_TYPE_PROCESS: PROCESS,
    DM_TYPE_SYSTEM: SYSTEM
    }

DM_DESCRIPTION = "Description"
DM_ACCEPTNEWPROPERTY = "AcceptNewProperty"
DM_PROPERTYLIST ="PropertyList"
DM_BASECLASS = "Baseclass"
DM_BUILTIN = "Builtin"

DM_PROPERTY_DEFAULTVALUE = "DEFAULTVALUE"
DM_PROPERTY_SETTABLE_FLAG = "SETTABLEFLAG"
DM_PROPERTY_GETTABLE_FLAG = "GETTABLEFLAG"
DM_PROPERTY_SAVEABLE_FLAG = "SAVEABLEFLAG"
DM_PROPERTY_LOADABLE_FLAG = "LOADABLEFLAG"
DM_PROPERTY_DELETEABLE_FLAG = "DELETEABLEFLAG"
DM_PROPERTY_TYPE = "TYPE"

DM_PROP_TYPE_STRING     = "String"
DM_PROP_TYPE_POLYMORPH  = "Polymorph"
DM_PROP_TYPE_INTEGER    = "Integer"
DM_PROP_TYPE_REAL       = "Real"

DM_SYSTEM_CLASS_OLD = 'CompartmentSystem'
DM_SYSTEM_CLASS = 'System'
DM_VARIABLE_CLASS = 'Variable'

DM_CAN_INSTANTIATE = 0
DM_CAN_LOADINFO = 1
DM_CAN_ADDPROPERTY = 2

DM_ATTR_BUILTIN           = 1 << 0
DM_ATTR_INSTANTIABLE      = 1 << 1
DM_ATTR_INFO_LOADABLE     = 1 << 2
DM_ATTR_DYN_PROPERTY_SLOT = 1 << 3

DM_PROP_ATTR_SETTABLE     = 1 << 0
DM_PROP_ATTR_GETTABLE     = 1 << 1
DM_PROP_ATTR_LOADABLE     = 1 << 2
DM_PROP_ATTR_SAVEABLE     = 1 << 3
DM_PROP_ATTR_DYNAMIC      = 1 << 4

DMINFO_STEPPER_SYSTEMLIST = 'SystemList'
DMINFO_STEPPER_PROCESSLIST = 'ProcessList'
DMINFO_PROCESS_VARREFLIST = 'VariableReferenceList'
DMINFO_ENTITY_STEPPERID = 'StepperID'
DMINFO_SYSTEM_STEPPERID = DMINFO_ENTITY_STEPPERID
DMINFO_SYSTEM_SIZE = 'Size'
DMINFO_PROCESS_STEPPERID = DMINFO_ENTITY_STEPPERID

N_A = 6.0221367e+23
