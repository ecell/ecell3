#include "libecs.hpp"
#include "Process.hpp"


namespace libecs
{

  LIBECS_DM_CLASS( ContinuousProcess, Process )
  {

  public:

    LIBECS_DM_OBJECT_ABSTRACT( ContinuousProcess )
      {
	INHERIT_PROPERTIES( Process );
      }
  
    ContinuousProcess()
      {
	; // do nothing
      }
  
    virtual ~ContinuousProcess()
      {
	;
      }

    virtual const bool isContinuous() const
    {
      return true;
    }

  protected:
  
  };

}
