CLASSNAME = 'MichaelisUniUniProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','KmS',0.0), ('Real','KcF',0.0)]

PROTECTED_AUX = '''
  Connection S0;
  Connection C0;
'''

defineMethod( 'initialize', '''
  S0 = getConnection( "S0" );
  C0 = getConnection( "C0" );
''' )

defineMethod( 'process', '''
  Real velocity( KcF );

  velocity *= C0.getVariable()->getValue();
  const Real S( S0.getVariable()->getConcentration() );
  velocity *= S;
  velocity /= ( KmS + S );

  setFlux( velocity );
''' )
