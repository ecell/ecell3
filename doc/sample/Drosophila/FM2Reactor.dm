CLASSNAME = 'FM2Reactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','vm',0.0),('Real','Km',0.0)]

PROTECTED_AUX = '''
  Reactant P0;
'''

defineMethod( 'initialize', '''
  P0 = getReactant( "P0" );
''' )

defineMethod( 'react', '''
  Real E( P0.getSubstance()->getConcentration() );

  Real V( -1 * vm * E );
  V /= Km + E;
  V *= 1E-018 * N_A;

  process( V );
''' )
