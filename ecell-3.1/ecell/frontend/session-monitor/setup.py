#!/usr/bin/env python

import os
import sys
from glob import glob
from distutils.core import setup, Extension
import ecell.config as config

def relative( basePath, path ):
	separators = os.sep + ( os.altsep or "" )
	commonPrefix = os.path.commonprefix( [ basePath, path ] )
	if len( basePath[ len( commonPrefix ): ].rstrip( separators ) ) != 0:
		return path
	else:
		return path[ len( commonPrefix ): ].lstrip( separators )

resources = glob(
    os.path.join( os.path.dirname( __file__ ), 'glade', '*' ) )

plugins = glob(
    os.path.join( os.path.dirname( __file__ ), 'plugins', '*' ) )

setup(
    name = 'ecell.session-monitor',
    version = config.version,
    description = 'E-Cell Osogo -- Session Monitor',
    author = 'E-Cell project',
    author_email = 'info@e-cell.org',
    url = 'http://www.e-cell.org/',
    packages = [ 'ecell.ui', 'ecell.ui.osogo' ],
    scripts = [ 'ecell3-session-monitor' ],
    data_files = [
        ( relative( config.prefix, config.conf_dir ), [ 'osogo.ini' ] ),
        ( os.path.join( relative( config.prefix, config.data_dir ), 'session-monitor', 'glade' ), resources ),
        ( os.path.join( relative( config.prefix, config.lib_dir), 'session-monitor', 'plugins' ), plugins ),
        ]
    )
