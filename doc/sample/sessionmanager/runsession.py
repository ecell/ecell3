loadModel( MODEL_FILE )                # Load the model.

S = createEntityStub( 'Variable:/:S' ) # Create a stub object of 

S[ 'Value' ] = VALUE_OF_S              # Set the value VALUE_OF_S given by the 
                                       # ESM script.

run( 200 )                             # Run the simulation for 200 seconds.

message( S[ 'Value' ] )                # Print the value of 'Variable:/:S'.

