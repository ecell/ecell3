CLASSNAME = 'FM2Process'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','vm',0.0),('Real','Km',0.0)]

PROTECTED_AUX = '''
  Connection P0;
'''

defineMethod( 'initialize', '''
  P0 = getConnection( "P0" );
''' )

defineMethod( 'process', '''
  Real E( P0.getVariable()->getConcentration() );

  Real V( -1 * vm * E );
  V /= Km + E;
  V *= 1E-018 * N_A;

  setFlux( V );
''' )
