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



from ConfirmWindow import *
import os
SHAPE_PLUGIN_PATH=os.environ['MEPATH']+os.sep+"plugin"+os.sep

ME_DESIGN_MODE = "Design"
ME_RUN_MODE = "Run"



# Layout constants
#Shape Plugin Constants
SHAPE_PLUGIN_NAME='Default Shape'
SHAPE_PLUGIN_TYPE='Default Type'

DEFAULT_SHAPE_NAME='Default'

# object types:
LE_OBJECT_SYSTEM = 0
LE_OBJECT_PROCESS = 1
LE_OBJECT_VARIABLE = 2
LE_OBJECT_TEXTBOX = 3
LE_OBJECT_CUSTOM = 4
LE_OBJECT_CONNECTION = 5

# Default Root System Stepper
DE_DEFAULT_STEPPER_CLASS = 'ODEStepper'
DE_DEFAULT_STEPPER_NAME = 'Default_Stepper'
DE_DEFAULT_PROCESS_CLASS = 'ExpressionFluxProcess'

#CL_PATH='ME_PATH'
# ModelStore constants

MS_STEPPER_CLASS = 0
MS_STEPPER_PROPERTYLIST = 1
MS_STEPPER_INFO = 2
MS_STEPPER_SYSTEMLIST = 'SystemList'
MS_STEPPER_PROCESSLIST = 'ProcessList'
MS_VARIABLE_PROCESSLIST = 'ProcessList'
MS_PROCESS_VARREFLIST = 'VariableReferenceList'

MS_PROPERTY_VALUE = 0
MS_PROPERTY_FLAGS = 1
MS_PROPERTY_TYPE = 2
MS_PROPERTY_COLOR = 3

#MS_SETTABLE_FLAG = 0
#MS_GETTABLE_FLAG = 1

AVOGADRO = 6.0221367e+23



MS_SETTABLE_FLAG = 0
MS_GETTABLE_FLAG = 1
MS_LOADABLE_FLAG = 2
MS_SAVEABLE_FLAG = 3
MS_DELETEABLE_FLAG = 4
MS_CHANGED_FLAG = 5

MS_VARREF_NAME = 0
MS_VARREF_FULLID = 1
MS_VARREF_COEF = 2

MS_VARIABLE_VALUE = "Value"
MS_VARIABLE_NUMCONC = "NumberConc"
MS_VARIABLE_MOLARCONC = "MolarConc"
MS_SIZE = "SIZE"

MS_SYSTEM_STEPPERID = 'StepperID'
MS_PROCESS_STEPPERID = 'StepperID'
DND_PROPERTYLIST_TYPE = "propertylist"
DND_PROPERTYVALUELIST_TYPE = "propertyvaluelist"

MS_ENTITY_CLASS = 0
MS_ENTITY_PROPERTYLIST = 1
MS_ENTITY_PARENT = 2
MS_ENTITY_CHILD_SYSTEMLIST = 3
MS_ENTITY_CHILD_PROCESSLIST = 4
MS_ENTITY_CHILD_VARIABLELIST = 5
MS_ENTITY_INFO = 6

MS_SYSTEM_ROOT = 'System::/'



# DM constants
ECELL_PROPERTY_SETTABLE_FLAG = 0

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
DM_PROPERTY_LOADABLE_FLAG = "LOADABLEFLAG"
DM_PROPERTY_SAVEABLE_FLAG = "SAVEABLEFLAG"

DM_PROPERTY_TYPE = "TYPE"

DM_PROPERTY_STRING = "STRING"
DM_PROPERTY_MULTILINE = "MULTILINE"
DM_PROPERTY_NESTEDLIST = "NESTED_LIST"
DM_PROPERTY_INTEGER = "INTEGER"
DM_PROPERTY_FLOAT = "FLOAT"

DM_SYSTEM_CLASS_OLD = 'CompartmentSystem'
DM_SYSTEM_CLASS = 'System'

DM_VARIABLE_CLASS = 'Variable'


DM_CAN_INSTANTIATE = 0
DM_CAN_LOADINFO = 1
DM_CAN_ADDPROPERTY = 2

# message types
ME_PLAINMESSAGE = 0
ME_OKCANCEL = 1
ME_YESNO = 2
ME_WARNING = 3
ME_ERROR = 4
ME_STATUSBAR = 5

ME_RESULT_OK = OK_PRESSED
ME_RESULT_CANCEL = CANCEL_PRESSED

# propertyattributes
ME_GETTABLE_FLAG = MS_GETTABLE_FLAG
ME_SETTABLE_FLAG = MS_SETTABLE_FLAG
ME_LOADABLE_FLAG = MS_LOADABLE_FLAG
ME_SAVEABLE_FLAG = MS_SAVEABLE_FLAG
ME_DELETEABLE_FLAG = MS_DELETEABLE_FLAG
ME_CHANGED_FLAG = MS_CHANGED_FLAG

ME_ROOTID = 'System::/'

# varref indexes
ME_VARREF_NAME = 0
ME_VARREF_FULLID = 1
ME_VARREF_COEF = 2

# entity types
ME_SYSTEM_TYPE = "System"
ME_VARIABLE_TYPE = "Variable"
ME_PROCESS_TYPE = "Process"
ME_STEPPER_TYPE = "Stepper"
ME_PROPERTY_TYPE = "Property"

# special properties
ME_STEPPER_SYSTEMLIST = MS_STEPPER_SYSTEMLIST
ME_STEPPER_PROCESSLIST = MS_STEPPER_PROCESSLIST
ME_VARIABLE_PROCESSLIST = MS_VARIABLE_PROCESSLIST
ME_PROCESS_VARREFLIST = MS_PROCESS_VARREFLIST
ME_STEPPERID = MS_SYSTEM_STEPPERID

#special editor type
ME_ENTITY_EDITOR = 'ME_ENTITY_EDITOR'
ME_CONNNECTION_OBJ_EDITOR = 'ME_CONNNECTION_OBJ_EDITOR'

# undo
MAX_REDOABLE_COMMAND = 100

# ADCP flags
ME_ADD_FLAG = 0
ME_DELETE_FLAG = 1
ME_COPY_FLAG = 2
ME_PASTE_FLAG = 3
ME_EDIT_FLAG = 4
ME_BROWSE_FLAG = 5
ME_APPEND_FLAG = 6

ME_FLAGS_NO = 7

# varrefs
ME_VARREF_FULLID = 1

### PATHWAYEDITOR CONSTANTS
# direction
DIRECTION_UP = 4
DIRECTION_DOWN = 8
DIRECTION_LEFT = 1
DIRECTION_RIGHT = 2
DIRECTION_BOTTOM_RIGHT=10
DIRECTION_BOTTOM_LEFT=9
DIRECTION_TOP_RIGHT=6
DIRECTION_TOP_LEFT=5

# CURSOR types
CU_POINTER = 0
CU_MOVE = 1
CU_ADD = 2
CU_RESIZE_TOP_LEFT = 3
CU_RESIZE_TOP = 4
CU_RESIZE_TOP_RIGHT = 5
CU_RESIZE_RIGHT = 6
CU_RESIZE_BOTTOM_RIGHT = 7
CU_RESIZE_BOTTOM = 8
CU_RESIZE_BOTTOM_LEFT = 9
CU_RESIZE_LEFT = 10
CU_CONN_INIT = 11
CU_CROSS=12

#4- up, 5, down, 

# object types
OB_TYPE_PROCESS = "Process"
OB_TYPE_VARIABLE = "Variable"
OB_TYPE_SYSTEM = "System"
OB_TYPE_TEXT = "Text"
OB_TYPE_CONNECTION = "Connection"
OB_NOTHING = "NOTHING"
OB_SHOW_LABEL=1
OB_MIN_WIDTH = 80
OB_MIN_HEIGTH = 40

# object properties
OB_POS_X = 'x'   #omitted
OB_POS_Y = 'y'   #omitted
OB_FULLID = 'FullID'  #rename action
OB_LABEL = 'Label'
OB_MINLABEL='Min Label Length'
OB_STEPPERID = 'StepperID' #omitted
OB_TYPE = 'Type' #omitted
OB_DIMENSION_X = 'DimensionX' #omitted in Editor Object, resize action in SystemObject
OB_DIMENSION_Y = 'DimensionY' #omitted in Editor Object, resize action in SystemObject
OB_HASFULLID = 'HasFullID' #omitted

OB_OUTLINE_COLOR = "OutlineColor" #outline change action
OB_FILL_COLOR = "FillColor"  #fill change action
OB_TEXT_COLOR = "TextColor" # textcolot change action
OB_OUTLINE_WIDTH = "Outline" #outline change action

OB_SHAPE_TYPE = "ShapeType" #shapetype change action
OB_SHAPEDESCRIPTORLIST = "ShapeDescriptorList" # cannot change: omitted



# connection constants
# directions
PROCESS_TO_VARIABLE = 0
VARIABLE_TO_PROCESS = 1

# connection properties
CO_PROCESS_ATTACHED = "ProcessAttached"  # this is an ID or None! #omitted
CO_VARIABLE_ATTACHED = "VariableAttached" # this is an ID or None! #omitted
CO_PROCESS_RING = "ProcessRing"  #omitted
CO_VARIABLE_RING = "VariableRing" #omitted
CO_NAME = "Name" #rename action
CO_COEF = "Coefficient" # change coef action
CO_ISRELATIVE = "Isrelative" # omitted
# lower level connection properties
CO_LINETYPE = "LineType"  # change linetype action
CO_LINEWIDTH = "LineWidth" # change linewidth
CO_CONTROL_POINTS = "ControlPoints"    
CO_ENDPOINT1 = "Endpoint1" # omitted
CO_ENDPOINT2 = "Endpoint2" # omitted
CO_DIRECTION1 = "Direction1" #omitted
CO_DIRECTION2 = "Direction2" #omitted
CO_HASARROW1 = "Hasarrow1" #change arrow action
CO_HASARROW2 = "Hasarrow2" #change arrow action
CO_ATTACHMENT1TYPE = "Attachment1Type" # attached thing to endpoint 1 ( Process, Variable, Nothing ) #omitted
CO_ATTACHMENT2TYPE = "Attachment2Type" #omitted

CO_USEROVERRIDEARROW = "UserSetArrow" #omitted

# process properties
PR_CONNECTIONLIST = "ConnectionList" #omitted

# variable properties
VR_CONNECTIONLIST = "ConnectionList" #omitted


# system properties
SY_INSIDE_DIMENSION_X = "InsideDimensionX"  #omitted
SY_INSIDE_DIMENSION_Y = "InsideDimensionY" #omitted
SY_PASTE_CONNECTIONLIST = "PasteConnectionList" #omitted

# selector variable
PE_SELECTOR = "Selector"
PE_VARIABLE ="Variable"
PE_PROCESS = "Process"
PE_SYSTEM= "System"
PE_CUSTOM = "Custom"
PE_TEXT =  "Text"


# local/global properties

GLOBALPROPERTYSET = [ CO_COEF, CO_NAME, OB_SHAPE_TYPE, CO_LINETYPE ]
MODELPROPERTYSET = [ CO_COEF, CO_NAME ]

### CANVAS CONSTANTS

# shape types
SHAPE_TYPE_SQUARE = "Square"
SHAPE_TYPE_VARIABLE = "Variable"
SHAPE_TYPE_PROCESS = "Process"
SHAPE_TYPE_SYSTEM = "System"
SHAPE_TYPE_TEXTBOX = "TextBox"
SHAPE_TYPE_CUSTOM = "Custom"
SHAPE_TYPE_STRAIGHT_LINE = "Straight"
SHAPE_TYPE_CORNERED_LINE = "Cornered"
SHAPE_TYPE_CURVED_LINE = "Curved"
SHAPE_TYPE_MULTIBCURVE_LINE = "MultiBezierCurve"

# layout properties
LO_SCROLL_REGION = "ScrollRegion" # list of int
LO_ZOOM_RATIO ="Zoom ratio"
LO_ROOT_SYSTEM = "Rootsystem"

#SHAPEDESCRIPTOR properties
SD_NAME = 0
SD_TYPE = 1
SD_FUNCTION = 2
SD_COLOR = 3
SD_Z = 4
SD_SPECIFIC = 5
SD_PROPERTIES = 6


#specific descriptors
RECT_RELX1 = 0
RECT_RELY1 = 1
RECT_RELX2 = 2
RECT_RELY2 = 3
RECT_ABSX1 = 4
RECT_ABSY1 = 5
RECT_ABSX2 = 6
RECT_ABSY2 = 7
#for rect. line, ellipse specific
SPEC_POINTS=0
SPEC_WIDTH_RATIO = 1
SPEC_LABEL = 1

LINE_POINTS = 0 # [ x1abs, x1rel, y1abs, y1rel,... ] for connection lines it x1, y1, x2, y2
LINE_WIDTH = 1

TEXT_TEXT = 0
TEXT_RELX = 1
TEXT_RELY = 2
TEXT_ABSX = 3
TEXT_ABSY = 4

BPATH_PATHDEF=0
BPATH_WIDTH = 1

# parameters for SD_FUNCTION and SD_COLOR
SD_OUTLINE = 0
SD_FILL = 1
SD_TEXT = 2
SD_RING = 3 # initates connectionobject by mousedrag
SD_NONE = 4 # does nothing by mousedrag
SD_SYSTEM_CANVAS = 5
SD_ARROWHEAD = 6
SD_FIXEDLINE = 7
SD_MOVINGLINE = 8

IMG_FILENAME = 1

#gnomecanvasobjects:
CV_RECT = "CanvasRect"
CV_ELL = "CanvasEllipse"
CV_TEXT = "CanvasText"
CV_LINE = "CanvasLine"
CV_BPATH = "CanvasBPath"
CV_IMG = "CanvasWidget"

# parameters for minimum SYSTEM_TYPE dimensions
SYS_MINWIDTH=230
SYS_MINHEIGHT=200
SYS_MINLABEL=29

# parameters for minimum VARIABLE_TYPE dimensions
VAR_MINWIDTH=10
VAR_MINHEIGHT=10
VAR_MINLABEL=5

# parameters for minimum PROCESS_TYPE dimensions
PRO_MINWIDTH=10
PRO_MINHEIGHT=10
PRO_MINLABEL=5

# parameters for minimum TEXT_TYPE dimensions
TEXT_MINWIDTH=205
TEXT_MINHEIGHT=27

# attachment points
RING_TOP = "RingTop"
RING_BOTTOM = "RingBottom"
RING_LEFT = "RingLeft"
RING_RIGHT = "RingRight"
RINGANGLES = { RING_TOP:90, RING_BOTTOM :270, RING_LEFT:180, RING_RIGHT:0 }
# line parts
ARROWHEAD1 = "arrowhead1"
ARROWHEAD2 = "arrowhead2"
EXTEND1 = "extendline1"
EXTEND2 = "extendline2"
ARROWHEAD_LENGTH = 10



