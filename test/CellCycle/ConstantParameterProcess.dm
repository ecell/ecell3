
CLASSNAME = 'ConstantParameterProcess'
BASECLASS = 'Process'
PROPERTIES = []

PROTECTED_AUX = '''
  Real _value;
'''

defineMethod( 'initialize', '''
//
''' )

defineMethod( 'process', '''
//  _value = getSuperSystem()->getStepper()->getStepInterval();
  _value = getStepper()->getStepInterval();
  setActivity( _value );
''' )

