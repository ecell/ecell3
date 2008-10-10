#!/usr/bin/env python

import os
import sys
from glob import glob
from distutils.core import setup, Extension
import ecell.config as config

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
        ( config.conf_dir, [ 'osogo.ini' ] ),
        ( os.path.join( config.data_dir, 'session-monitor', 'glade' ), resources ),
        ( os.path.join( config.lib_dir, 'session-monitor', 'plugins' ), plugins ),
        ]
    )
