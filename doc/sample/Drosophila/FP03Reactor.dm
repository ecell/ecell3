CLASSNAME = 'FP03Reactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','V2',0.0),('Real','K2',0.0)]

PROTECTED_AUX = '''
  Reactant C0;
'''

defineMethod( 'initialize', '''
  C0 = getReactant( "C0" );
''' )

defineMethod( 'react', '''
  Real E( C0.getSubstance()->getConcentration() );

  Real V( V2 * E );
  V /= K2 + E;
  V *= 1E-018 * N_A;

  process( V );
''' )
