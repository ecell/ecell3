CLASSNAME = 'MassActionProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','k',0.0)]

PROTECTED_AUX ='''
'''

defineMethod( 'initialize','''
''' )

defineMethod( 'react',
'''
  Real velocity( k * N_A );
  velocity *= getSuperSystem()->getVolume();
  
  for( VariableReferenceListIterator i( theVariableReferenceList.begin() );
       i != theVariableReferenceList.end(); ++i )
    {
      VariableReference aVariableReference( *i );
      Int aCoefficient( aVariableReference.getCoefficient() );
      if( aCoefficient < 0 )
        {
          do{
            aCoefficient++;
            velocity *= aVariableReference.getVariable()->getConcentration();
          } while( aCoefficient != 0 );
        }
     }
  setFlux(velocity);
''')

