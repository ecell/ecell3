CLASSNAME = 'MassActionReactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','k',0.0)]

PROTECTED_AUX ='''
  Reactant P0;
'''

defineMethod( 'initialize','''
''' )

defineMethod( 'react',
'''
  Real velocity( k * N_A );
  velocity *= getSuperSystem()->getVolume();
  
  for( ReactantMapIterator s( theReactantMap.begin() );
       s != theReactantMap.end(); ++s )
    {
      Reactant aReactant( s->second );
      Int aStoichiometry( aReactant.getStoichiometry() );
      if( aStoichiometry < 0 )
        {
          do{
            aStoichiometry++;
            velocity *= aReactant.getSubstance()->getConcentration();
          }while(aStoichiometry != 0 );
        }
     }
  process(velocity);
''')

