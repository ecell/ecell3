
# A simple heat shock model to demonstrate how to construct stochastic and 
# deterministic composite model using ECELL 3.
#     July 8, 2003

Stepper ODE45Stepper( DE1 )
{
        # no property
}

@#{MAIN_STEPPER='DE1'}
@#{MAIN_PROCESS='MassActionFluxProcess'}
@{HYBRID_STEPPER='DE1'}
@{HYBRID_PROCESS='MassActionFluxProcess'}

@{MAIN_STEPPER='NR1'}
@{MAIN_PROCESS='GillespieProcess'}
@#{HYBRID_STEPPER='NR1'}
@#{HYBRID_PROCESS='GillespieProcess'}

Stepper NRStepper( NR1 ){Tolerance      0;}

@{
DNAJ = 310.
UNPROTEIN = 200000.

PROTEIN = 5000000.
DNAJ_UNPROTEIN = 5000000.

UNFOLD = 2e-1

UNPROTEIN_BINDING = PROTEIN / DNAJ / UNPROTEIN * 6.03e8 * UNFOLD
REFOLD = PROTEIN / DNAJ_UNPROTEIN * UNFOLD * 1.0

}

System System( / )
{
        StepperID       @MAIN_STEPPER;

        Variable PositiveVariable( SIZE )
        {
                Value   1.5e-15;
        }

        Variable PositiveVariable( DNAS32 )
        {
                Value   1;
        }
        
        Variable PositiveVariable( mRNAS32 )
        {
                Value   17;
        }
        
        Variable PositiveVariable( S32 )
        {
                Value   15;
        }


        Variable PositiveVariable( mRNAS32_Nothing )
        {
                Value   0;
        }

        Variable PositiveVariable( E_S32 )
        {
                Value   76;
        }
        
        Variable PositiveVariable( DNADnaJ )
        {
                Value   1;
        }


        Variable PositiveVariable( DnaJ )
        {
                Value 464;
        }

        Variable PositiveVariable( DnaJ_Nothing )
        {
                Value 0;
        }

        Variable PositiveVariable( S32_DnaJ )
        {
                Value   2959;
        }
        
        Variable PositiveVariable( DNAFtsH )
        {
                Value   0;
        }
        
        Variable PositiveVariable( FtsH )
        {
                Value   200;
        }

        Variable PositiveVariable( FtsH_Nothing )
        {
                Value   0;
        }

        Variable PositiveVariable( DNAGroEL )
        {
                Value   1;
        }
        
        Variable PositiveVariable( GroEL )
        {
                Value   4314;
        }
        
        Variable PositiveVariable( GroEL_Nothing )
        {
                Value   0;
        }

        Variable PositiveVariable( Protein )
        {
                Value   @(PROTEIN);
        }

        Variable PositiveVariable( UnProtein )
        {
                Value   @(UNPROTEIN);
        }

        Variable PositiveVariable( DnaJ_UnProtein )
        {
                Value   @(DNAJ_UNPROTEIN);
        }

        #################################################

        #Reaction stochastic part

        ################################################
        
        Process @(MAIN_PROCESS)( S32_transcription )
        {
                VariableReferenceList   [ S0 Variable:/:DNAS32 -1 ] 
                                        [ P0 Variable:/:mRNAS32 1 ]
                                        [ P1 Variable:/:DNAS32 1];
                k       1.4e-3;
        }
        
        Process @(MAIN_PROCESS)( S32_translation )
        {
                VariableReferenceList   [ S0 Variable:/:mRNAS32 -1 ]
                                        [ P0 Variable:/:S32 1 ] 
                                        [ P1 Variable:/:mRNAS32 1];
                k       0.07;
        
        }
        
        Process @(MAIN_PROCESS)( mRNAS32_degradation )
        {
                VariableReferenceList   [ S0 Variable:/:mRNAS32 -1 ] 
                                        [ P0 Variable:/:mRNAS32_Nothing 1 ];
                k       1.4e-6;
        
        }

        Process @(MAIN_PROCESS)( E_S32_association )
        {
                VariableReferenceList   [ S0 Variable:/:S32 -1 ]
                                        [ P0 Variable:/:E_S32 1];
                k       0.7;
        }
        
        Process @(MAIN_PROCESS)( E_S32_dissociation )
        {
                VariableReferenceList   [ S0 Variable:/:E_S32 -1 ] 
                                        [ P0 Variable:/:S32 1 ];
                k       0.13;
        }
        
        Process @(MAIN_PROCESS)( DnaJ_expression )
        {
                VariableReferenceList   [ S0 Variable:/:DNADnaJ -1 ]
                                        [ S1 Variable:/:E_S32 -1]
                                        [ P0 Variable:/:DnaJ 1 ]
                                        [ P1 Variable:/:DNADnaJ 1]
                                        [ P2 Variable:/:S32 1];
                k       4.41e6;
        }

        Process @(MAIN_PROCESS)( DnaJ_degradation )
        {
                VariableReferenceList   [ S0 Variable:/:DnaJ -1 ]
                                        [ P0 Variable:/:DnaJ_Nothing 1];
                k       6.4e-10;
        
        }
        
        Process @(MAIN_PROCESS)( S32_DnaJ_association )
        {
                VariableReferenceList   [ S0 Variable:/:S32 -1 ]
                                        [ S1 Variable:/:DnaJ -1 ]
                                        [ P0 Variable:/:S32_DnaJ 1];
                k       3.27e5;
        }
        
        Process @(MAIN_PROCESS)( S32_DnaJ_dissociation )
        {
                VariableReferenceList   [ S0 Variable:/:S32_DnaJ -1 ]
                                        [ P0 Variable:/:S32 1 ]
                                        [ P1 Variable:/:DnaJ 1];
                k       4.4e-4;
        }

        Process @(MAIN_PROCESS)( FtsH_expression )
        {
                VariableReferenceList   [ S0 Variable:/:DNAFtsH -1 ]
                                        [ S1 Variable:/:E_S32 -1]
                                        [ P0 Variable:/:FtsH 1 ]
                                        [ P1 Variable:/:DNAFtsH 1]
                                        [ P2 Variable:/:S32 1];
                k       4.41e6;
        }
        
        Process @(MAIN_PROCESS)( FtsH_degradation )
        {
                VariableReferenceList   [ S0 Variable:/:FtsH -1 ]
                                        [ P0 Variable:/:FtsH_Nothing 1 ];
                k       7.4e-11;#
        }
        
        Process @(MAIN_PROCESS)( S32_degradation )
        {
                VariableReferenceList   [ S0 Variable:/:S32_DnaJ -1 ]
                                        [ S1 Variable:/:FtsH -1]
                                        [ P0 Variable:/:DnaJ 1 ]
                                        [P1 Variable:/:FtsH 1];
                k       1.28e3;#
        }

        Process @(MAIN_PROCESS)( GroEL_expression )
        {
                VariableReferenceList   [ S0 Variable:/:DNAGroEL -1 ]
                                        [ S1 Variable:/:E_S32 -1]
                                        [ P0 Variable:/:GroEL 1 ]
                                        [ P1 Variable:/:DNAGroEL 1]
                                        [ P2 Variable:/:S32 1];
                k       5.69e6;
        }
        
        Process @(MAIN_PROCESS)( GroEL_degradation )
        {
                VariableReferenceList   [ S0 Variable:/:GroEL -1 ]
                                        [ P0 Variable:/:GroEL_Nothing 1 ];
                k       1.8e-8;
        }

        ########################################

        #Reaction deterministic part

        ########################################

        Process @(HYBRID_PROCESS)( Unfold )
        {
                StepperID @HYBRID_STEPPER;
                VariableReferenceList   [ S0 Variable:/:Protein -1]
                                        [ P0 Variable:/:UnProtein 1];
                k       @(UNFOLD);
        }

        Process @(HYBRID_PROCESS)( UnProteinBinding )
        {
                StepperID @HYBRID_STEPPER;
                VariableReferenceList   [S0 Variable:/:DnaJ -1]
                                        [S1 Variable:/:UnProtein -1]
                                        [P0 Variable:/:DnaJ_UnProtein 1]; 
                k       @(UNPROTEIN_BINDING);
        }

        Process @(HYBRID_PROCESS)( Refold )
        {
                StepperID @HYBRID_STEPPER;
                VariableReferenceList   [S0 Variable:/:DnaJ_UnProtein -1]
                                        [P0 Variable:/:Protein 1] 
                                        [P1 Variable:/:DnaJ 1];
                k       @(REFOLD);
        }
        

}

