CLASSNAME = 'FP22Process'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','V4',0.0),('Real','K4',0.0)]

PROTECTED_AUX = '''
  VariableReference C0;
'''

defineMethod( 'initialize', '''
  C0 = getVariableReference( "C0" );
''' )

defineMethod( 'process', '''
  Real E( C0.getConcentration() );

  Real V( -1 * V4 * E );
  V /= K4 + E;
  V *= 1E-018 * N_A;

  setFlux( V );
''' )
