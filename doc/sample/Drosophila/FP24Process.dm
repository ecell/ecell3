CLASSNAME = 'FP24Process'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','k2',0.0)]

PROTECTED_AUX = '''
  VariableReference C0;
'''

defineMethod( 'initialize', '''
  C0 = getVariableReference( "C0" );
''' )

defineMethod( 'process', '''
  Real E( C0.getConcentration() );

  Real V( k2 * E );
  V *= 1E-018 * N_A;

  setFlux( V );
''' )
