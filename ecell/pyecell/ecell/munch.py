#!/usr/bin/python
#
# $Id$ $Date$

"""
A system for processing Python as markup embedded in text.
"""

VERSION = '0.3'
AUTHOR = 'Erik Max Francis <max@alcyone.com>'
COPYRIGHT = 'Copyright (C) 2002 Erik Max Francis'
LICENSE = 'GPL'


import string
import StringIO
import sys
import types


PREFIX = '@'
FAILURE_CODE = 1


class MunchError(Exception):
    """The base class for all Munch errors."""
    pass

class FreezeError(MunchError):
    """An error in freezing or unfreezing has taken place."""
    pass

class DiversionError(MunchError):
    """An invalid diversion was accessed."""
    pass

class ParseError(MunchError):
    """A parse error occurred."""
    pass

class TransientParseError(ParseError):
    """A parse error occurred which may be resolved by feeding more data.
    Such an error reaching the toplevel is an unexpected EOF error."""
    pass

class StackUnderflowError(MunchError):
    """A stack underflow."""
    pass


class MetaError(Exception):
    """A wrapper around a real Python exception for including a copy of
    the context."""
    
    def __init__(self, contexts, exc):
        Exception.__init__(self, exc)
        self.contexts = contexts
        self.exc = exc

    def __str__(self):
        backtrace = map(lambda x: "%s:%d" % (x.name, x.line), self.contexts)
        return "%s: %s (%s)" % (self.exc.__class__, self.exc, \
                                (string.join(backtrace, ', ')))


class ProxyFile:
    """A wrapper around an (output) file object which supports diversions."""
    
    def __init__(self, file):
        self.file = file
        self.currentDiversion = None
        self.diversions = {}

    def write(self, data):
        if self.currentDiversion is None:
            self.file.write(data)
        else:
            self.diversions[self.currentDiversion].write(data)
    
    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def divert(self, diversion=None):
        self.currentDiversion = diversion
        if diversion is not None and not self.diversions.has_key(diversion):
            self.diversions[diversion] = StringIO.StringIO()

    def undivert(self, diversion=None):
        if diversion is not None:
            # Undivert a particular diversion.
            if self.diversions.has_key(diversion):
                strFile = self.diversions[diversion]
                self.file.write(strFile.getvalue())
                del self.diversions[diversion]
                if self.currentDiversion == diversion:
                    self.currentDiversion = None
            else:
                raise DiversionError, "nonexistent diversion: %s" % diversion
        else:
            # Undivert all pending diversions.
            if self.diversions:
                diversions = self.diversions.keys()
                diversions.sort()
                for diversion in diversions:
                    strFile = self.diversions[diversion]
                    self.file.write(strFile.getvalue())
                self.diversions = {}
                self.currentDiversion = None
            

    def close(self):
        self.undivert()
        self.file.close()


class Parser:
    """The core parser which also manages the output file."""
    
    IDENTIFIER_FIRST = '_abcdefghijklmnopqrstuvwxyz' + \
                       'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    IDENTIFIER_REST = '0123456789' + IDENTIFIER_FIRST + '.'

    def __init__(self, proxy, globals, locals=None):
        self.proxy = proxy
        self.buffer = ''
        self.globals = globals
        self.locals = locals

    def process(self, data):
        """Process data in the form of a 3-triple:  the data preceding the
        token, the token, and the data following the token (which may
        include more tokens).  If no token is found, then it will return
        data, None, ''."""
        loc = string.find(data, PREFIX)
        if loc < 0:
            # No token, end of data.
            return data, None, ''
        else:
            prefix, suffix = string.split(data, PREFIX, 1)
            if not suffix:
                raise ParseError, "illegal trailing token"
            elif suffix[0] in PREFIX + "#({` \t\v\n":
                code, suffix = suffix[0], suffix[1:]
                return prefix, PREFIX + code, suffix
            elif suffix[0] in Parser.IDENTIFIER_FIRST:
                return prefix, PREFIX, suffix
            else:
                raise ParseError, "unrecognized token sequence"

    def scanNext(self, data, i, target):
        """Scan from i for the next occurrence of the target character, 
        respecting string literals."""
        quote = None
        dataLen = len(data)
        while i < dataLen:
            c = data[i]
            if c == "'" or c == '"':
                if not quote:
                    quote = c
                elif quote == c:
                    quote = None
            elif quote:
                if c == '\\':
                    i = i + 1
            elif not quote:
                if c == target:
                    return i
            i = i + 1
        else:
            raise TransientParseError, "expecting ending character"

    def scanComplex(self, data, i, enter, exit):
        """Scan from i for an ending sequence, respecting entries and exits,
        respecting string literals."""
        quote = None
        depth = 0
        dataLen = len(data)
        while i < dataLen:
            c = data[i]
            if c == "'" or c == '"':
                if not quote:
                    quote = c
                elif quote == c:
                    quote = None
            elif quote:
                if c == '\\':
                    i = i + 1
            elif not quote:
                if c == enter:
                    depth = depth + 1
                elif c == exit:
                    depth = depth - 1
                    if depth < 0:
                        return i
            i = i + 1
        else:
            raise TransientParseError, "expecting end of complex expression"

    def scanWord(self, data, i=0):
        """Scan from i for a simple word."""
        dataLen = len(data)
        while i < dataLen:
            if not data[i] in Parser.IDENTIFIER_REST:
                return i
            i = i + 1
        else:
            raise TransientParseError, "expecting end of word"

    def scanPhrase(self, data, i=0):
        """Scan from i for a phrase (e.g., 'word', 'f(a, b, c)', 'a[i]', or
        combinations like 'x[i](a)'."""
        # Find the word.
        i = self.scanWord(data, i)
        while i < len(data) and data[i] in '([':
            enter = data[i]
            exit = {'(': ')', '[': ']',}[enter]
            i = self.scanComplex(data, i + 1, enter, exit, ) + 1
        return i
    
    def scanSimple(self, data, i=0):
        """Scan from i for a simple expression, which consists of one 
        more phrases separated by dots."""
        i = self.scanPhrase(data, i)
        dataLen = len(data)
        while i < dataLen and data[i] == '.':
            i = self.scanPhrase(data, i)
        # Make sure we don't end with a trailing dot.
        while i > 0 and data[i - 1] == '.':
            i = i - 1
        return i

    def feed(self, data):
        """Feed the parser more data."""
        self.buffer = self.buffer + data

    def once(self):
        """Feed the parser more data (if that argument is presented)
        and have it attempt to parse it by only performing _one_
        expansion.  If a parse error occurs which may be eliminated by
        parsing more data, a TransientParseError is raised; catch and
        dispose of this error if you intend to send more data.  Return
        true if an expansion was performed, otherwise false."""
        if self.buffer:
            prefix, token, self.buffer = self.process(self.buffer)
            self.proxy.write(prefix)
            if token is None:
                # No remaining tokens, just pass through.
                self.proxy.write(self.buffer)
                self.buffer = ''
                return 0
            elif token in (PREFIX + ' ', PREFIX + '\t', \
                           PREFIX + '\v', PREFIX + '\n'):
                # Whitespace/line continuation.
                pass
            elif token == PREFIX:
                # "Simple expression" expansion.
                try:
                    i = self.scanSimple(self.buffer)
                    code, self.buffer = self.buffer[:i], self.buffer[i:]
                    result = self.evaluate(code)
                    if result is not None:
                        self.proxy.write(str(result))
                except TransientParseError:
                    self.buffer = token + self.buffer
                    raise
            elif token == PREFIX + '#':
                # Comment.
                if string.find(self.buffer, '\n') >= 0:
                    self.buffer = string.split(self.buffer, '\n', 1)[1]
                else:
                    self.buffer = token + self.buffer
                    raise TransientParseError, "expecting newline"
            elif token == PREFIX + '(':
                # Expression evaluation.
                try:
                    i = self.scanComplex(self.buffer, 0, '(', ')')
                    code, self.buffer = self.buffer[:i], self.buffer[i + 1:]
                    result = self.evaluate(code)
                    if result is not None:
                        self.proxy.write(str(result))
                except TransientParseError:
                    self.buffer = token + self.buffer
                    raise
            elif token == PREFIX + '`':
                # Repr evalulation.
                try:
                    i = self.scanNext(self.buffer, 0, '`')
                    code, self.buffer = self.buffer[:i], self.buffer[i + 1:]
                    self.proxy.write(repr(self.evaluate(code)))
                except TransientParseError:
                    self.buffer = token + self.buffer
                    raise
            elif token == PREFIX + '{':
                # Statement evaluation.
                try:
                    i = self.scanComplex(self.buffer, 0, '{', '}')
                    code, self.buffer = self.buffer[:i], self.buffer[i + 1:]
                    self.execute(code)
                except TransientParseError:
                    self.buffer = token + self.buffer
                    raise
            elif token == PREFIX * 2:
                self.proxy.write(PREFIX)
            else:
                raise ParseError, "unknown token: %s" % token
            return 1
        else:
            return 0

    def parse(self):
        while self.buffer:
            self.once()

    def done(self):
        """Declare that this parsing session is over and that any unparsed
        data should be considered an error."""
        if self.buffer:
            if self.buffer[-1] != '\n':
                # The data is not well-formed text; add a newline.
                self.buffer = self.buffer + '\n'
            # A TransientParseError thrown here is a real parse error.
            self.parse()
            self.buffer = ''

    def evaluate(self, expression):
        """Evaluate an expression."""
        if self.locals:
            return eval(expression, self.globals, self.locals)
        else:
            return eval(expression, self.globals)

    def execute(self, statements):
        """Execute a statement."""
        if string.find(statements, '\n') < 0:
            statements = string.strip(statements)
        if self.locals:
            exec statements in self.globals, self.locals
        else:
            exec statements in self.globals

    def __nonzero__(self): return len(self.buffer) != 0


class PseudoModule:
    """A pseudomodule for the builtin munch routines."""

    pass


class Context:
    """An interpreter context, which encapsulates a name, an input file
    object, and a parser object."""

    def __init__(self, name, input, parser, line=0):
        self.name = name
        self.input = input
        self.parser = parser
        self.line = line
        self.exhausted = 1

    def __str__(self):
        return '%s:%d' % (self.name, self.line) ###


class Stack:
    """A simple stack that behave as a sequence (with 0 being the top
    of the stack, not the bottom)."""

    def __init__(self, seq=None):
        if seq is None:
            seq = []
        self.data = seq

    def top(self):
        try:
            return self.data[-1]
        except IndexError:
            raise StackUnderflowError, "stack is empty for top"
        
    def pop(self):
        try:
            return self.data.pop()
        except IndexError:
            raise StackUnderflowError, "stack is empty for pop"
        
    def push(self, object): self.data.append(object)

    def clone(self):
        """Create a duplicate of this stack."""
        return self.__class__(self.data[:])

    def __nonzero__(self): return len(self.data) != 0
    def __len__(self): return len(self.data)
    def __getitem__(self, index): return self.data[-(index + 1)]


class Interpreter:
    """Higher-level manipulation of a stack of parsers, with file/line
    context."""

    def __init__(self, output=None, globals=None):
        # Set up the proxy first, since .pseudo needs it.
        if output is None:
            output = sys.stdout
        self.proxy = ProxyFile(output)
        if globals is None:
            globals = {'munch': self.pseudo()}
        sys.stdout = self.proxy ### A better way?
        self.globals = globals
        self.contexts = Stack()

    def pseudo(self):
        """Construct and return a proper 'munch' pseudomodule, already
        bound to the helper methods in this parser."""
        module = PseudoModule()
        module.identify = self.__identify
        module.expand = self.__expand
        module.include = self.__include
        module.divert = self.proxy.divert
        module.undivert = self.proxy.undivert
        return module

    def new(self, name, input, locals=None):
        """Create a new interpreter context (a new file object to process)."""
        parser = Parser(self.proxy, self.globals, locals)
        context = Context(name, input, parser)
        self.contexts.push(context)

    def step(self):
        """Read one line of the current input interactively (attempt
        to parse it); if no input remains, finalize the context and
        pop it.  Return true if code was successfully parsed."""
        context = self.contexts.top()
        if context.exhausted:
            # If the current context is exhausted, get then next line.
            line = context.input.readline()
            context.line = context.line + 1
            if not line:
                context.parser.done()
                self.contexts.pop()
                return 0
            context.parser.feed(line)
            context.exhausted = 0
        try:
            # If the context changes after an expansion, abort this step.
            while context.parser and context is self.contexts.top():
                context.parser.once()
            if not context.parser:
                context.exhausted = 1
        except TransientParseError:
            context.exhausted = 1
        return 1

    def go(self):
        """Read the whole file."""
        while self.contexts:
            self.step()

    def finish(self):
        """Declare this interpreting session over; close the proxy file
        object."""
        if self.proxy is not None:
            self.proxy.close()
            self.proxy = None

    def meta(self, exc=None):
        """Construct a MetaError for the interpreter's current state."""
        return MetaError(self.contexts.clone(), exc)

    def handle(self, meta):
        """Handle a MetaError."""
        first = 1
        for context in meta.contexts:
            if first:
                if meta.exc is not None:
                    desc = "error: %s: %s" % (meta.exc.__class__, meta.exc)
                else:
                    desc = "error"
            else:
                desc = "from this context"
            first = 0
            sys.stderr.write('%s:%s: %s\n' % \
                             (context.name, context.line, desc))

    def __identify(self):
        """Identify the topmost context with a 2-tuple of the name and the
        line number."""
        context = self.contexts.top()
        return context.name, context.line

    def __expand(self, data, locals=None):
        """Do an explicit expansion pass on a string."""
        file = StringIO.StringIO(data)
        self.new('<expand>', file, locals)

    def __include(self, fileOrFilename, locals=None):
        """Take another filename or file object and process it."""
        if type(fileOrFilename) is types.StringType:
            # Either it's a string ...
            filename = fileOrFilename
            name = filename
            file = open(filename, 'r')
        else:
            # ... or a file object.
            file = fileOrFilename
            name = "<%s>" % str(file.__class__)
        self.new(name, file, locals)


def usage():
    def warn(message):
        sys.stderr.write(message + '\n')
    warn("Usage: %s [options] <filenames, or '-' for stdin>..." % sys.argv[0])
    warn("")
    warn("Valid options:")
    warn("  -h --help                    print usage")
    warn("  -p --prefix <char>           change prefix to something other than @")
    warn("  -r --raw --raw-errors        show raw Python errors")
    warn("  -o --output <filename>       specify file for output")
    warn("")
    warn("The following expansions are supported (where @ is the prefix):")
    warn("  @# comment newline           comment; remove everything up to newline")
    warn("  @ whitespace                 remove following whitespace; line continuation")
    warn("  @@                           literal @; @ is escaped (duplicated prefix)")
    warn("  @(expression)                evaluate expression and substitute str")
    warn("  @simple_expression           evaluate simple expression and substitute;")
    warn("                               e.g., @x, @x.y, @f(a, b), @l[i], etc.")
    warn("  @{statements}                statements are executed for side effects")
    warn("  @`expression`                evaluate expression and substitute repr")
    warn("")
    warn("Notes:")
    warn("Whitespace immediately inside parentheses of @(...) are ignored.  Whitespace")
    warn("immediately inside braces of @{...} are ignored, unless ... spans multiple")
    warn("lines.  Use @{ ... }@ to suppress newline following expansion.")
    warn("Simple expressions ignore trailing dots; '@x.' means '@(x).'.")

def main():
    args = sys.argv[1:]
    args.reverse()
    Output = None
    Globals = None
    RawErrors = 0
    while args and args[-1][0] == '-' and args[-1] != '-':
        option = args.pop()
        if option == '--':
            break
        elif option == '-h' or option == '--help':
            usage()
            return
        elif option == '-p' or option == '--prefix':
            char = args.pop()
            if type(char) != types.StringType and len(char) != 1:
                raise ValueError, "new prefix must be a single character string"
            global PREFIX
            PREFIX = char
        elif option == '-r' or option == '--raw' or option == '--raw-errors':
            RawErrors = 1
        elif option == '-o' or option == '--option':
            filename = args.pop()
            Output = open(filename, 'w')
        else:
            raise ValueError, "unknown option: %s" % option
    if not args:
        args.append('-')
    args.reverse()
    interpreter = Interpreter(Output, Globals)
    try:
        for filename in args:
            if filename == '-':
                name = '<stdin>'
                file = sys.stdin
            else:
                name = filename
                file = open(filename, 'r')
            interpreter.new(name, file)
            try:
                interpreter.go()
            except Exception, e:
                meta = interpreter.meta(e)
                interpreter.handle(meta)
                if RawErrors:
                    raise
                sys.exit(FAILURE_CODE)
    finally:
        interpreter.finish()

if __name__ == '__main__': main()
