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
  void resizeScreen(unsigned, unsigned);
  void setStep(char* buffer);
  void setTime(char* buffer);
  void setXangle(double);
  void setYangle(double);
  void setZangle(double);
protected:
  bool isChanging;
  bool on_background_clicked(GdkEventButton*);
  bool on_checkbutton_clicked(GdkEventButton*, unsigned int);
  unsigned int theSpeciesSize;
  void onResetBound();
  void onResetRotation();
  void on_3DMolecule_toggled();
  void on_InvertBound_toggled();
  void on_checkbutton_toggled(unsigned int id);
  void on_record_toggled();
  void on_resetTime_clicked();
  void on_showSurface_toggled();
  void on_showTime_toggled();
  void screenChanged();
  void update_background_color(Gtk::ColorSelection*);
  void update_species_color(unsigned int, Gtk::ColorSelection*);
  void xLowBoundChanged();
  void xRotateChanged();
  void xUpBoundChanged();
  void yLowBoundChanged();
  void yRotateChanged();
  void yUpBoundChanged();
  void zLowBoundChanged();
  void zRotateChanged();
  void zUpBoundChanged();
protected:
  GLScene* m_area;
  Gtk::CheckButton** theButtonList;
  Gtk::Label** theLabelList;
  Gtk::Table m_table;
  Gtk::Table* m_areaTable;
private:
  Gdk::Color theBgColor;
  Glib::RefPtr<Gtk::SizeGroup> m_sizeGroup;
  Gtk::Adjustment theDepthAdj;
  Gtk::Adjustment theHeightAdj;
  Gtk::Adjustment theWidthAdj;
  Gtk::Adjustment theXAdj;
  Gtk::Adjustment theXLowBoundAdj;
  Gtk::Adjustment theXUpBoundAdj;
  Gtk::Adjustment theYAdj;
  Gtk::Adjustment theYLowBoundAdj;
  Gtk::Adjustment theYUpBoundAdj;
  Gtk::Adjustment theZAdj;
  Gtk::Adjustment theZLowBoundAdj;
  Gtk::Adjustment theZUpBoundAdj;
  Gtk::Button theButtonResetTime;
  Gtk::Button theResetBoundButton;
  Gtk::Button theResetDepthButton;
  Gtk::Button theResetRotButton;
  Gtk::CheckButton theCheck3DMolecule;
  Gtk::CheckButton theCheckFix;
  Gtk::CheckButton theCheckInvertBound;
  Gtk::CheckButton theCheckShowSurface;
  Gtk::CheckButton theCheckShowTime;
  Gtk::Entry m_steps;
  Gtk::Entry m_time;
  Gtk::Frame theFrameBoundAdj;
  Gtk::Frame theFrameLatticeAdj;
  Gtk::Frame theFrameRotAdj;
  Gtk::Frame theFrameScreen;
  Gtk::HBox m_rightBox;
  Gtk::HBox m_stepBox;
  Gtk::HBox m_timeBox;
  Gtk::HBox the3DMoleculeBox;
  Gtk::HBox theBoxBoundFixReset;
  Gtk::HBox theBoxRotFixReset;
  Gtk::HBox theDepthBox;
  Gtk::HBox theHeightBox;
  Gtk::HBox theWidthBox;
  Gtk::HBox theXBox;
  Gtk::HBox theXLowBoundBox;
  Gtk::HBox theXUpBoundBox;
  Gtk::HBox theYBox;
  Gtk::HBox theYLowBoundBox;
  Gtk::HBox theYUpBoundBox;
  Gtk::HBox theZBox;
  Gtk::HBox theZLowBoundBox;
  Gtk::HBox theZUpBoundBox;
  Gtk::HScale theDepthScale;
  Gtk::HScale theHeightScale;
  Gtk::HScale theWidthScale;
  Gtk::HScale theXLowBoundScale;
  Gtk::HScale theXScale;
  Gtk::HScale theXUpBoundScale;
  Gtk::HScale theYLowBoundScale;
  Gtk::HScale theYScale;
  Gtk::HScale theYUpBoundScale;
  Gtk::HScale theZLowBoundScale;
  Gtk::HScale theZScale;
  Gtk::HScale theZUpBoundScale;
  Gtk::Label m_bgColor;
  Gtk::Label m_stepLabel;
  Gtk::Label m_timeLabel;
  Gtk::Label theDepthLabel;
  Gtk::Label theHeightLabel;
  Gtk::Label theWidthLabel;
  Gtk::Label theXLabel;
  Gtk::Label theXLowBoundLabel;
  Gtk::Label theXUpBoundLabel;
  Gtk::Label theYLabel;
  Gtk::Label theYLowBoundLabel;
  Gtk::Label theYUpBoundLabel;
  Gtk::Label theZLabel;
  Gtk::Label theZLowBoundLabel;
  Gtk::Label theZUpBoundLabel;
  Gtk::SpinButton theDepthSpin;
  Gtk::SpinButton theHeightSpin;
  Gtk::SpinButton theWidthSpin;
  Gtk::SpinButton theXLowBoundSpin;
  Gtk::SpinButton theXSpin;
  Gtk::SpinButton theXUpBoundSpin;
  Gtk::SpinButton theYLowBoundSpin;
  Gtk::SpinButton theYSpin;
  Gtk::SpinButton theYUpBoundSpin;
  Gtk::SpinButton theZLowBoundSpin;
  Gtk::SpinButton theZSpin;
  Gtk::SpinButton theZUpBoundSpin;
  Gtk::ToggleButton m_3d;
  Gtk::ToggleButton m_showSurface;
  Gtk::ToggleButton m_showTime;
  Gtk::ToggleButton theButtonRecord;
  Gtk::VBox theBoxCtrl;
  Gtk::VBox theBoxInBound;
  Gtk::VBox theBoxInFrame;
  Gtk::VBox theBoxInLattice;
  Gtk::VBox theBoxInScreen;
};

class GLScene : public Gtk::GL::DrawingArea
{
public:
  Color getColor(unsigned int i) { return theSpeciesColor[i]; };
  Color getSpeciesColor(unsigned int id);
  GLScene(const Glib::RefPtr<const Gdk::GL::Config>& config,
          const char* aFileName);
  bool getSpeciesVisibility(unsigned int id);
  bool writePng();
  char* getSpeciesName(unsigned int id);
  static const unsigned int TIMEOUT_INTERVAL;
  unsigned int getColSize() { return theColSize; };
  unsigned int getLayerSize() { return theLayerSize; };
  unsigned int getRowSize() { return theRowSize; };
  unsigned int getSpeciesSize() { return theTotalSpeciesSize; };
  virtual ~GLScene();
  void drawTime();
  void invalidate() { get_window()->invalidate_rect(get_allocation(), false); }
  void pause();
  void play();
  void renderLayout(Glib::RefPtr<Pango::Layout>);
  void resetBound();
  void resetTime();
  void resetView();
  void rotate(int aMult, int x, int y, int z);
  void rotateMidAxis(int aMult, int x, int y, int z);
  void rotateMidAxisAbs(double, int , int , int );
  void set3DMolecule(bool is3D);
  void setInvertBound(bool);
  void setBackgroundColor(Color);
  void setControlBox(ControlBox* aControl);
  void setRecord(bool isRecord);
  void setReverse(bool isReverse);
  void setScreenHeight( unsigned int );
  void setScreenWidth( unsigned int );
  void setShowSurface(bool);
  void setShowTime(bool);
  void setSpeciesColor(unsigned int id, Color);
  void setSpeciesVisibility(unsigned int id, bool isVisible);
  void setXLowBound( unsigned int aBound );
  void setXUpBound( unsigned int aBound );
  void setYLowBound( unsigned int aBound );
  void setYUpBound( unsigned int aBound );
  void setZLowBound( unsigned int aBound );
  void setZUpBound( unsigned int aBound );
  void step();
  void translate(int x, int y, int z);
  void update() { get_window()->process_updates(false); }
  void zoomIn();
  void zoomOut();
protected:
  bool (GLScene::*theLoadCoordsFunction)(std::streampos&);
  bool loadCoords(std::streampos&);
  bool loadMeanCoords(std::streampos&);
  virtual bool on_configure_event(GdkEventConfigure* event);
  virtual bool on_expose_event(GdkEventExpose* event);
  virtual bool on_map_event(GdkEventAny* event);
  virtual bool on_timeout();
  virtual bool on_unmap_event(GdkEventAny* event);
  virtual bool on_visibility_notify_event(GdkEventVisibility* event);
  virtual void on_realize();
  void (GLScene::*thePlot3DFunction)();
  void (GLScene::*thePlotFunction)();
  void drawBox(GLfloat xlo, GLfloat xhi, GLfloat ylo, GLfloat yhi, GLfloat zlo,
               GLfloat zhi);
  void drawScene(double);
  void normalizeAngle(double&);
  void plot3DCubicMolecules();
  void plot3DHCPMolecules();
  void plotCubicPoints();
  void plotGrid();
  void plotHCPPoints();
  void plotMean3DCubicMolecules();
  void plotMean3DHCPMolecules();
  void plotMeanHCPPoints();
  void setColor(unsigned int i, Color *c);
  void setLayerColor(unsigned int i);
  void setRandColor(Color *c);
  void setTranslucentColor(unsigned int i, GLfloat j);
  void timeout_add();
  void timeout_remove();
protected:
  Color* theSpeciesColor;
  ControlBox* m_control;
  GLfloat Aspect;
  GLfloat FieldOfView;
  GLfloat Near;
  GLfloat ViewMidx;
  GLfloat ViewMidy;
  GLfloat ViewMidz;
  GLfloat ViewSize;
  GLfloat X;
  GLfloat Xtrans;
  GLfloat Y;
  GLfloat Ytrans;
  GLfloat Z;
  GLfloat prevX;
  GLfloat prevY;
  GLfloat prevZ;
  GLfloat theBCCc;
  GLfloat theHCPl;
  GLfloat theHCPx;
  GLfloat theHCPy;
  GLfloat theRotateAngle;
  GLuint m_FontListBase;
  Glib::RefPtr<Pango::Context> ft2_context;
  Glib::ustring m_FontString;
  Glib::ustring m_timeString;
  Point* theMeanPoints;
  Point** thePoints;
  bool *theSpeciesVisibility;
  bool isChanged;
  bool isInvertBound;
  bool m_Run;
  bool m_RunReverse;
  bool show3DMolecule;
  bool showSurface;
  bool showTime;
  bool startRecord;
  char** theSpeciesNameList;
  double *theRadii;
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
  int m_FontHeight;
  int m_FontWidth;
  int m_stepCnt;
  int theGLIndex;
  sigc::connection m_ConnectionTimeout;
  std::ifstream theFile;
  std::map<unsigned int, Point> theCoordPoints;
  std::size_t font_size;
  std::size_t pixel_extent_height;
  std::size_t pixel_extent_width;
  std::size_t tex_height;
  std::size_t tex_width;
  std::vector<std::streampos> theStreamPosList;
  std::vector<unsigned int> thePolySpeciesList;
  unsigned int theColSize;
  unsigned int theCutCol;
  unsigned int theCutLayer;
  unsigned int theCutRow;
  unsigned int theDimension;
  unsigned int theLatticeSpSize;
  unsigned int theLatticeType;
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
  unsigned int* theMoleculeSize;
  unsigned int* theOffLatticeMoleculeSize;
  unsigned int* theXLowBound;
  unsigned int* theXUpBound;
  unsigned int* theYLowBound;
  unsigned int* theYUpBound;
  unsigned int* theZLowBound;
  unsigned int* theZUpBound;
  unsigned int** theCoords;
  unsigned int** theFrequency;
};

class Rulers : public Gtk::Window
{
public:
  Rulers(const Glib::RefPtr<const Gdk::GL::Config>& config,
         const char* aFileName);
protected:
  //signal handlers:
  //Gtk::DrawingArea m_area;
  GLScene m_area;
  Gtk::HPaned m_hbox;
  Gtk::HRuler m_hrule;
  Gtk::Table m_table;
  Gtk::VRuler m_vrule;
  ControlBox m_control;
  bool isRecord;
  static const int XSIZE = 250, YSIZE = 250;
  virtual bool on_area_motion_notify_event(GdkEventMotion* event); //override
  virtual bool on_expose(GdkEventExpose* event);
  virtual bool on_key_press_event(GdkEventKey* event);
};

#endif /* __SpatiocyteVisualizer_hpp */

