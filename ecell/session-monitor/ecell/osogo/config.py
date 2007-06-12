import sys
import os
from ecell.config import *

package_local = 'session-monitor'
package = package + '.' + package_local

for key in ( 'data_dir', 'lib_dir' ):
    locals()[ key ] = os.path.join( locals()[ key ], package_local )

# where to place .glade files for main (non-plugin) windows
glade_dir = os.path.join( data_dir, 'glade' )

# where to look at when loading plug-ins
plugin_path = [ os.path.join( lib_dir, 'plugins' ) ]
