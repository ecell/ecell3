CLASSNAME = 'FP25Process'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','vd',0.0),('Real','Kd',0.0)]

PROTECTED_AUX = '''
  Connection C0;
'''

defineMethod( 'initialize', '''
  C0 = getConnection( "C0" );
''' )

defineMethod( 'react', '''
  Real E( C0.getVariable()->getConcentration() );

  Real V( -1 * vd * E );
  V /= Kd + E;
  V *= 1E-018 * N_A;

  process( V );
''' )
