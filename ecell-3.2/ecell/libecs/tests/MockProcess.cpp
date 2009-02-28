LIBECS_DM_CLASS( MockProcess, Process )
{
public:
    LIBECS_DM_OBJECT( MockProcess, Process )
    {
        INHERIT_PROPERTIES( Process );
    }

    virtual void fire() {}

    virtual void initialize()
    {
        Process::initialize();
    }    
};

LIBECS_DM_INIT_STATIC( MockProcess, Process );
