
class LayoutCommand( Command ):


	def __checkArgs( self ):
		if type (self._Command__theReceiver) == 'object':
			if self._Command__theReceiver.__class__.__name__ == 'Layout':
				return True

		return False





createLayout
deleteLayout
renameLayout
cloneLayout
pastelayout
copylayout

createobject
deleteobject
changeobjectProperty
copyobject
cutobject
pasteobject
moveobject
resize object

createConnection
redirectConnection


