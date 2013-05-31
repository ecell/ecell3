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
#
# Designed by Koichi Takahashi <shafi@e-cell.org>
# Programmed by Masahiro Sugimoto <msugi@sfc.keio.ac.jp>

ECELL3_PYTHON = 'ecell3-python'
ECELL3_SESSION = 'ecell3-session'

DEFAULT_STDOUT = 'stdout'
DEFAULT_STDERR = 'stderr'

SYSTEM_PROXY = 'SystemProxy'
SESSION_PROXY = 'SessionProxy'

DEFAULT_TMP_DIRECTORY = 'tmp'
DEFAULT_ENVIRONMENT = 'Local'

# job status
QUEUED       = 0
RUN          = 1
FINISHED     = 2
ERROR        = 3

STATUS = {
    QUEUED:   'QUEUED',
    RUN:      'RUN',
    FINISHED: 'FINISHED',
    ERROR:    'ERROR',
    }

INTERESTING_ENV_VARS = (
    'ECELL3_PREFIX',
    'ECELL3_DM_PATH',
    'LTDL_LIBRARY_PATH',
    'LD_LIBRARY_PATH',
    'PYTHONPATH',
    'PATH', # XXX: better not include this one?
    )
