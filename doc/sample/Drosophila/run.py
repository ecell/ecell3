loadModel( 'Drosophila.eml' )

a = createEntityStub( 'Variable:/CELL/CYTOPLASM:P0' )
#s = createStepperStub( 'DE' )


#print s.getProperty( 'ReadVariableList' )
#print s.getProperty( 'SystemList' )
print a.getProperty( 'Value' )
run( 10 )

print a.getProperty( 'Value' )
