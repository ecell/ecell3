CLASSNAME = 'FP14Process'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','V4',0.0),('Real','K4',0.0)]

PROTECTED_AUX = '''
  Connection C0;
'''

defineMethod( 'initialize', '''
  C0 = getConnection( "C0" );
''' )

defineMethod( 'process', '''
  Real E( C0.getVariable()->getConcentration() );

  Real V( V4 * E );
  V /= K4 + E;
  V *= 1E-018 * N_A;

  setFlux( V );
''' )
