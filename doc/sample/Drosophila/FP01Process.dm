CLASSNAME = 'FP01Process'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','Km',0.0)]

PROTECTED_AUX = '''
  Connection C0;
'''

defineMethod( 'initialize', '''
  C0 = getConnection( "C0" );
''' )

defineMethod( 'react', '''
  Real E( C0.getVariable()->getConcentration() );

  Real V( Km * E );
  V *= 1E-018 * N_A;

  process( V );
''' )
