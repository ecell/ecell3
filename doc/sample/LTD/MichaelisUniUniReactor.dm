CLASSNAME = 'MichaelisUniUniReactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','Km',0.0), ('Real','Kcat',0.0)]

PROTECTED_AUX = '''
  Reactant S0;
  Reactant C0;
'''

defineMethod( 'initialize', '''
  S0 = getReactant( "S0" );
  C0 = getReactant( "C0" );
''' )

defineMethod( 'react', '''
  Real velocity( Kcat );

  velocity *= C0.getSubstance()->getQuantity();
  const Real S( S0.getSubstance()->getConcentration() );
  velocity *= S;
  velocity /= ( Km + S );

  process( velocity );
''' )

