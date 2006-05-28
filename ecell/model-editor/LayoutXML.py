from xml.dom import minidom

LXML_DOC_STRING = '<sbml xmlns="http://www.sbml.org/sbml/level2" level="2" version="1"\
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\
      xmlns:sl2="http://projects.eml.org/bcb/sbml/level2"\
      xsi:schemaLocation="http://projects.eml.org/bcb/sbml/level2 \
               http://projects.eml.org/bcb/sbml/level2/layout2.xsd">\
        <model></model><annotation></annotation></sbml>'

LXML_LAYOUTLIST = 'sl2:listOfLayouts'
LXML_LAYOUT = 'sl2:layout'
LXML_COMPARTMENTLIST = "sl2:listOfCompartmentGlyphs"
LXML_COMPARTMENT = "sl2:compartmentGlyph"
LXML_SPECIES = "sl2:speciesGlyph"
LXML_SPECIESLIST = "sl2:listOfSpeciesGlyphs"
LXML_REACTION = "sl2:reactionGlyph"
LXML_REACTIONLIST = "sl2:listOfReactionGlyphs"
LXML_SPECIESREFERENCE = "sl2:speciesReferenceGlyph"
LXML_SPECIESREFERENCELIST = "sl2:listOfSpeciesReferenceGlyphs"
LXML_CURVESEGMENTLIST="sl2:/listOfCurveSegments"
LXML_CURVESEGMENT="sl2:curveSegment"


class LayoutXML:

    def __init__( self, aText=None ):
        if aText==None:
            self.theDocument = minidom.parseString(DOC_STRING)
        else:
            self.theDocument = minidom.parseString( aText)


    def getEmlName( self ):
        pass


    def setEmlName( self, aName ):
        pass


    def getLayoutList( self ):
        pass


    def createLayout( self, aLayoutName ):
        pass


    def getLayoutProperty( self, aLayoutName, aPropertyName):
        pass


    def setLayoutProperty( self, aLayoutName, aPropertyName, aPropertyValue ):
        pass

    def createObject( self, aLayoutName, aObjectName, aObjectType='SHAPE'):
        pass

    def getObjectList( self, aLayoutName, aObjectType='SHAPE'):
        pass


    def getObjectProperty( self, aLayoutName, aObjectName, aPropertyName, aObjectType ='SHAPE'):
        pass

    def setObjectProperty( self, aLayoutName, aObjectName, aPropertyName, aPropertyValue, aObjectType ='SHAPE'):
        pass

