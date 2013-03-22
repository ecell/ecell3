//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2006-2009 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Satya Arjunan <satya.arjunan@gmail.com>
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//


#ifndef __SpatiocyteVisualizer_hpp
#define __SpatiocyteVisualizer_hpp

#define SPHERE          1
#define BOX             2
#define GRID            3

//Lattice type:
#define HCP_LATTICE   0
#define CUBIC_LATTICE 1

using namespace std;

struct Color
{
  float r;
  float g;
  float b;
};

struct Point 
{
  double  x;
  double  y;
  double  z;
};

class GLScene;

class ControlBox : public Gtk::ScrolledWindow
{
public:
  ControlBox(GLScene*, Gtk::Table*);
  virtual ~ControlBox();
  void setStep(char* buffer);
  void setTime(char* buffer);
  void setXangle(double);
  void setYangle(double);
  void setZangle(double);
  void resizeScreen(unsigned, unsigned);
protected:
  void on_checkbutton_toggled(unsigned int id);
  bool on_checkbutton_clicked(GdkEventButton*, unsigned int);
  bool on_background_clicked(GdkEventButton*);
  void update_species_color(unsigned int, Gtk::ColorSelection*);
  void update_background_color(Gtk::ColorSelection*);
  void on_3DMolecule_toggled();
  void on_showTime_toggled();
  void on_showSurface_toggled();
  void on_record_toggled();
  void on_resetTime_clicked();
  void onResetRotation();
  void onResetBound();
  void xRotateChanged();
  void yRotateChanged();
  void zRotateChanged();
  void screenChanged();
  void xUpBoundChanged();
  void xLowBoundChanged();
  void yUpBoundChanged();
  void yLowBoundChanged();
  void zUpBoundChanged();
  void zLowBoundChanged();
  Gtk::Table m_table;
  Gtk::CheckButton** theButtonList;
  Gtk::Label** theLabelList;
  unsigned int theSpeciesSize;
private:
  Gtk::Entry m_steps;
  Gtk::Entry m_time;
  Gtk::HBox m_stepBox;
  Gtk::HBox m_timeBox;
  Gtk::HBox m_rightBox;
  Gtk::VBox theBoxCtrl;
  Gtk::Frame theFrameRotAdj;
  Gtk::VBox theBoxInFrame;
  Gtk::ToggleButton m_3d;
  Gtk::ToggleButton m_showTime;
  Gtk::ToggleButton m_showSurface;
  Gtk::Label m_stepLabel;
  Gtk::Label m_timeLabel;
  Gtk::HBox theXBox;
  Gtk::HBox theYBox;
  Gtk::HBox theZBox;
  Gtk::HBox theBoxRotFixReset;
  Gtk::Button theResetRotButton;
  Gtk::CheckButton theCheckFix;
  Gtk::Label theXLabel;
  Gtk::Adjustment theXAdj;
  Gtk::HScale theXScale;
  Gtk::SpinButton theXSpin;
  Gtk::Label theYLabel;
  Gtk::Adjustment theYAdj;
  Gtk::HScale theYScale;
  Gtk::SpinButton theYSpin;
  Gtk::Label theZLabel;
  Gtk::Adjustment theZAdj;
  Gtk::HScale theZScale;
  Gtk::SpinButton theZSpin;
  Gtk::Frame theFrameBoundAdj;
  Gtk::Frame theFrameScreen;
  Gtk::VBox theBoxInScreen;
  Gtk::HBox theHeightBox;
  Gtk::Label theHeightLabel;
  Gtk::Adjustment theHeightAdj;
  Gtk::HScale theHeightScale;
  Gtk::SpinButton theHeightSpin;
  Gtk::HBox theWidthBox;
  Gtk::Label theWidthLabel;
  Gtk::Adjustment theWidthAdj;
  Gtk::HScale theWidthScale;
  Gtk::SpinButton theWidthSpin;
  Gtk::VBox theBoxInBound;
  Gtk::HBox theXUpBoundBox;
  Gtk::HBox theXLowBoundBox;
  Gtk::HBox theYUpBoundBox;
  Gtk::HBox theYLowBoundBox;
  Gtk::HBox theZUpBoundBox;
  Gtk::HBox theZLowBoundBox;
  Gtk::HBox theBoxBoundFixReset;
  Gtk::Label m_bgColor;
  Gtk::CheckButton theCheckFixBound;
  Gtk::Button theResetBoundButton;
  Gtk::Label theXUpBoundLabel;
  Gtk::Adjustment theXUpBoundAdj;
  Gtk::HScale theXUpBoundScale;
  Gtk::SpinButton theXUpBoundSpin;
  Gtk::Label theXLowBoundLabel;
  Gtk::Adjustment theXLowBoundAdj;
  Gtk::HScale theXLowBoundScale;
  Gtk::SpinButton theXLowBoundSpin;
  Gtk::Label theYUpBoundLabel;
  Gtk::Adjustment theYUpBoundAdj;
  Gtk::HScale theYUpBoundScale;
  Gtk::SpinButton theYUpBoundSpin;
  Gtk::Label theYLowBoundLabel;
  Gtk::Adjustment theYLowBoundAdj;
  Gtk::HScale theYLowBoundScale;
  Gtk::SpinButton theYLowBoundSpin;
  Gtk::Label theZUpBoundLabel;
  Gtk::Adjustment theZUpBoundAdj;
  Gtk::HScale theZUpBoundScale;
  Gtk::SpinButton theZUpBoundSpin;
  Gtk::Label theZLowBoundLabel;
  Gtk::Adjustment theZLowBoundAdj;
  Gtk::HScale theZLowBoundScale;
  Gtk::SpinButton theZLowBoundSpin;
  Gtk::Frame theFrameLatticeAdj;
  Gtk::VBox theBoxInLattice;
  Gtk::HBox theDepthBox;
  Gtk::HBox the3DMoleculeBox;
  Gtk::Button theResetDepthButton;
  Gtk::CheckButton theCheck3DMolecule;
  Gtk::CheckButton theCheckShowTime;
  Gtk::CheckButton theCheckShowSurface;
  Gtk::Button theButtonResetTime;
  Gtk::Label theDepthLabel;
  Gtk::Adjustment theDepthAdj;
  Gtk::HScale theDepthScale;
  Gtk::SpinButton theDepthSpin;
  Gtk::ToggleButton theButtonRecord;
  Gdk::Color theBgColor;
  Glib::RefPtr<Gtk::SizeGroup> m_sizeGroup;
  Gtk::ColorButton m_Button;
protected:
  GLScene* m_area;
  Gtk::Table* m_areaTable;
  bool isChanging;
};

class GLScene : public Gtk::GL::DrawingArea
{
public:
  static const unsigned int TIMEOUT_INTERVAL;

public:
  GLScene(const Glib::RefPtr<const Gdk::GL::Config>& config,
          const char* aFileName);
  virtual ~GLScene();
protected:
  virtual void on_realize();
  virtual bool on_expose_event(GdkEventExpose* event);
  virtual bool on_map_event(GdkEventAny* event);
  virtual bool on_unmap_event(GdkEventAny* event);
  virtual bool on_visibility_notify_event(GdkEventVisibility* event);
  virtual bool on_timeout();
  virtual bool on_configure_event(GdkEventConfigure* event);
public:
  // Invalidate whole window.
  void rotate(int aMult, int x, int y, int z);
  void translate(int x, int y, int z);
  void rotateMidAxis(int aMult, int x, int y, int z);
  void rotateMidAxisAbs(double, int , int , int );
  void resetBound();
  void pause();
  void play();
  void resetView();
  void zoomIn();
  void zoomOut();
  bool writePng();
  void drawTime();
  void renderLayout(Glib::RefPtr<Pango::Layout>);
  void step();
  void setReverse(bool isReverse);
  void setSpeciesVisibility(unsigned int id, bool isVisible);
  bool getSpeciesVisibility(unsigned int id);
  void set3DMolecule(bool is3D);
  void setShowTime(bool);
  void setShowSurface(bool);
  void setRecord(bool isRecord);
  void resetTime();
  void setControlBox(ControlBox* aControl);
  Color getSpeciesColor(unsigned int id);
  void setSpeciesColor(unsigned int id, Color);
  void setBackgroundColor(Color);
  char* getSpeciesName(unsigned int id);
  void invalidate() {
    get_window()->invalidate_rect(get_allocation(), false);
  }

  // Update window synchronously (fast).
  void update()
  { get_window()->process_updates(false); }

  void setXUpBound( unsigned int aBound );
  void setXLowBound( unsigned int aBound );
  void setYUpBound( unsigned int aBound );
  void setYLowBound( unsigned int aBound );
  void setZUpBound( unsigned int aBound );
  void setZLowBound( unsigned int aBound );
  void setScreenWidth( unsigned int );
  void setScreenHeight( unsigned int );
  unsigned int getSpeciesSize()
    {
      return theTotalSpeciesSize;
    };
  unsigned int getLayerSize()
    {
      return theLayerSize;
    };
  unsigned int getColSize()
    {
      return theColSize;
    };
  unsigned int getRowSize()
    {
      return theRowSize;
    };
  Color getColor(unsigned int i)
    {
      return theSpeciesColor[i];
    };

protected:
  void drawBox(GLfloat xlo, GLfloat xhi, GLfloat ylo, GLfloat yhi,
                      GLfloat zlo, GLfloat zhi);
  void drawScene(double);
  void timeout_add();
  void plotGrid();
  void plot3DHCPMolecules();
  void plotMean3DHCPMolecules();
  void plotMeanHCPPoints();
  void plotHCPPoints();
  void plot3DCubicMolecules();
  void plotMean3DCubicMolecules();
  void plotCubicPoints();
  void timeout_remove();
  void setColor(unsigned int i, Color *c);
  void setRandColor(Color *c);
  void setTranslucentColor(unsigned int i, GLfloat j);
  void setLayerColor(unsigned int i);
  void (GLScene::*thePlotFunction)();
  void (GLScene::*thePlot3DFunction)();
  void normalizeAngle(double&);
  bool loadCoords(std::streampos&);
  bool loadMeanCoords(std::streampos&);
  bool (GLScene::*theLoadCoordsFunction)(std::streampos&);
protected:
  bool isChanged;
  bool m_Run;
  bool m_RunReverse;
  bool show3DMolecule;
  bool showTime;
  bool showSurface;
  bool startRecord;
  bool *theSpeciesVisibility;
  int m_FontHeight;
  int m_FontWidth;
  int theGLIndex;
  unsigned int m_stepCnt;
  unsigned int theCutCol;
  unsigned int theCutLayer;
  unsigned int theCutRow;
  unsigned int theColSize;
  unsigned int theDimension;
  unsigned int theLatticeType;
  unsigned int theLatticeSpSize;
  unsigned int theLayerSize;
  unsigned int theLogMarker;
  unsigned int theMeanCount;
  unsigned int theMeanPointSize;
  unsigned int theOffLatticeSpSize;
  unsigned int theOriCol;
  unsigned int theOriLayer;
  unsigned int theOriRow;
  unsigned int thePngNumber;
  unsigned int thePolymerSize;
  unsigned int theReservedSize;
  unsigned int theRowSize;
  unsigned int theScreenHeight;
  unsigned int theScreenWidth;
  unsigned int theStartCoord;
  unsigned int theThreadSize;
  unsigned int theTotalLatticeSpSize;
  unsigned int theTotalOffLatticeSpSize;
  unsigned int theTotalSpeciesSize;
  unsigned int* theXLowBound;
  unsigned int* theXUpBound;
  unsigned int* theYLowBound;
  unsigned int* theYUpBound;
  unsigned int* theZLowBound;
  unsigned int* theZUpBound;
  unsigned int* theMoleculeSize;
  unsigned int* theOffLatticeMoleculeSize;
  unsigned int** theCoords;
  unsigned int** theFrequency;
  double theCurrentTime;
  double theRadius;
  double theRealColSize;
  double theRealLayerSize;
  double theRealRowSize;
  double theResetTime;
  double theVoxelRadius;
  double xAngle;
  double yAngle;
  double zAngle;
  double *theRadii;
  char** theSpeciesNameList;
  GLuint m_FontListBase;
  GLfloat Aspect;
  GLfloat FieldOfView;
  GLfloat Near;
  GLfloat prevX;
  GLfloat prevY;
  GLfloat prevZ;
  GLfloat theBCCc;
  GLfloat theHCPl;
  GLfloat theHCPx;
  GLfloat theHCPy;
  GLfloat theRotateAngle;
  GLfloat ViewMidx;
  GLfloat ViewMidy;
  GLfloat ViewMidz;
  GLfloat ViewSize;
  GLfloat X;
  GLfloat Y;
  GLfloat Z;
  GLfloat Xtrans;
  GLfloat Ytrans;
  Point* theMeanPoints;
  Point** thePoints;
  Color* theSpeciesColor;
  ControlBox* m_control;
  sigc::connection m_ConnectionTimeout;
  Glib::ustring m_FontString;
  Glib::ustring m_timeString;
  std::ifstream theFile;
  std::vector<unsigned int> thePolySpeciesList;
  std::vector<std::streampos> theStreamPosList;
  std::map<unsigned int, Point> theCoordPoints;
  Glib::RefPtr<Pango::Context> ft2_context;
  std::size_t font_size;
  std::size_t pixel_extent_width;
  std::size_t pixel_extent_height;
  std::size_t tex_width;
  std::size_t tex_height;
};

class Rulers : public Gtk::Window
{
public:
  Rulers(const Glib::RefPtr<const Gdk::GL::Config>& config,
         const char* aFileName);

protected:

  //signal handlers:
  virtual bool on_area_motion_notify_event(GdkEventMotion* event); //override
  virtual bool on_key_press_event(GdkEventKey* event);
  virtual bool on_expose(GdkEventExpose* event);

  GLScene m_area;
  Gtk::Table m_table;
  Gtk::HPaned m_hbox;
  ControlBox m_control;
  //Gtk::DrawingArea m_area;
  Gtk::HRuler m_hrule;
  Gtk::VRuler m_vrule;
  static const int XSIZE = 250, YSIZE = 250;
  bool isRecord;
};

#endif /* __SpatiocyteVisualizer_hpp */

