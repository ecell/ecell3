CLASSNAME = 'FP14Reactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','V4',0.0),('Real','K4',0.0)]

PROTECTED_AUX = '''
  Reactant C0;
'''

defineMethod( 'initialize', '''
  C0 = getReactant( "C0" );
''' )

defineMethod( 'react', '''
  Real E( C0.getSubstance()->getConcentration() );

  Real V( V4 * E );
  V /= K4 + E;
  V *= 1E-018 * N_A;

  process( V );
''' )
