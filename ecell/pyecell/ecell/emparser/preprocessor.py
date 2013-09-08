#
# preprocessing methods
#

import StringIO

from ecell.emparser import em

class ecellHookClass(em.Hook):
    def __init__(self, aPreprocessor, anInterpreter):
        self.theInterpreter = anInterpreter
        self.thePreprocessor = aPreprocessor

    def afterInclude( self ):
        ( file, line ) = self.interpreter.context().identify()
        self.thePreprocessor.lineControl( self.theInterpreter, file, line )

    def beforeIncludeHook( self, name, file, locals ):  
        self.thePreprocessor.lineControl( self.theInterpreter, name, 1 )  

    def afterExpand( self, result ):
        self.thePreprocessor.need_linecontrol = 1

    def afterEvaluate(self, result):
        self.thePreprocessor.need_linecontrol = 1
        return

    def afterSignificate(self):
        self.thePreprocessor.need_linecontrol = 1
        return
                         
    def atParse(self, scanner, locals):
        if not self.thePreprocessor.need_linecontrol:
            return

        ( file, line ) = self.theInterpreter.context().identify()
        self.thePreprocessor.lineControl( self.theInterpreter, file, line )
        self.thePreprocessor.need_linecontrol = 0        

class Preprocessor( object ):

    def __init__( self, file, filename ):
        self.need_linecontrol = 0
        self.file = file
        self.filename = filename
        self.interpreter = None

    def __del__( self ):
        self.shutdown()

    def lineControl( self, interpreter, file, line ):
        interpreter.write( '%%line %d %s\n' % ( line, file ) )

    def needLineControl( self, *args ):
        self.need_linecontrol = 1

    def preprocess( self ):

        #
        # init
        #
        Output = StringIO.StringIO()
        self.interpreter = em.Interpreter( output = Output )
        self.interpreter.flatten()
        self.interpreter.addHook(ecellHookClass(self, self.interpreter))   # pseudo.addHook(ecellHookClass(self, self.interpreter))

        #
        # processing
        #

        # write first line
        self.lineControl( self.interpreter, self.filename, 1 )

        if self.file is not None:
            self.interpreter.wrap( self.interpreter.file,\
                      ( self.file, self.filename ) )

        self.interpreter.flush()

        return Output

    def shutdown( self ):
        
        self.interpreter.shutdown()



