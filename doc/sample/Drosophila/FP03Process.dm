CLASSNAME = 'FP03Process'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','V2',0.0),('Real','K2',0.0)]

PROTECTED_AUX = '''
  VariableReference C0;
'''

defineMethod( 'initialize', '''
  C0 = getVariableReference( "C0" );
''' )

defineMethod( 'process', '''
  Real E( C0.getVariable()->getConcentration() );

  Real V( V2 * E );
  V /= K2 + E;
  V *= 1E-018 * N_A;

  setFlux( V );
''' )
