CLASSNAME = 'FP13Process'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','V3',0.0),('Real','K3',0.0)]

PROTECTED_AUX = '''
  Connection C0;
'''

defineMethod( 'initialize', '''
  C0 = getConnection( "C0" );
''' )

defineMethod( 'process', '''
  Real E( C0.getVariable()->getConcentration() );

  Real V( -1 * V3 * E );
  V /= K3 + E;
  V *= 1E-018 * N_A;

  process( V );
''' )
