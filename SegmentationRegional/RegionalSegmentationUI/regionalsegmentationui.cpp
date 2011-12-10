#include "regionalsegmentationui.h"

#include <QtOpenCvWidget.h>
#include <RegionalSegmentationWorker.h>
#include <BSPlineLevelSet.h>
#include <algorithm>

using namespace cv;
using namespace Qwt3D;


Plot2D::Plot2D() : _levelSetFunction(this)
{
  setTitle("The Level Set Function Surface");
  setRotation(-30,0,15);
  setScale(1,1,1);
  setShift(0.15,0,0);
  setZoom(0.9);

  for (unsigned i=0; i!=coordinates()->axes.size(); ++i)
  {
	coordinates()->axes[i].setMajors(7);
	coordinates()->axes[i].setMinors(4);
  }

  coordinates()->axes[X1].setLabelString("X-axis");
  coordinates()->axes[Y1].setLabelString("Y-axis");
  coordinates()->axes[Z1].setLabelString( QChar (0X03A6)); // Phi - see http://www.unicode.org/charts/
  
  setCoordinateStyle(FRAME);      // NOCOORD FRAME  BOX
  this->setPlotStyle(FILLED);  // NOPLOT WIREFRAME HIDDENLINE FILLED FILLEDMESH POINTS USER
  setFloorStyle(FLOORISO);     //FLOORISO FLOORDATA
  this->setSmoothMesh(true);
}

void Plot2D::update( const cv::Mat* m, const double n )
{
  if (m)
  {
	_levelSetFunction._pM = m; 
	Size s = _levelSetFunction._pM->size();
	
	_levelSetFunction.setMesh(s.height,s.width);
	_levelSetFunction.setDomain(0,s.height-1,0,s.width-1);
	_levelSetFunction.setMinZ(- _levelSetFunction.scaleDrawFactor * (n + n/2));
	_levelSetFunction.setMaxZ(  _levelSetFunction.scaleDrawFactor * (n));
  }
  else
  {
	_levelSetFunction.setMesh(0,0);
	_levelSetFunction.setDomain(0,0,0,0);
	_levelSetFunction.setMinZ(0);
  }
  
  _levelSetFunction.create();
  updateData();
  updateGL();
  _levelSetFunction._pM = NULL; 
}

LevelSetFunctionSurface::LevelSetFunctionSurface( Qwt3D::SurfacePlot* pw ) : 
			Function(pw), _pM(NULL), scaleDrawFactor(10.0)
{
}

double LevelSetFunctionSurface::operator()( double x, double y )
{
  return scaleDrawFactor * _pM->at<double>(x,y);
}


RegionalSegmentationUI::RegionalSegmentationUI(QWidget *parent, Qt::WFlags flags)
	: QDialog(parent, flags), isMaskDrawing(false)     
{
  ui.setupUi(this);

  setWindowFlags( (windowFlags() | Qt::CustomizeWindowHint) & Qt::WindowMaximizeButtonHint); 

#define WINDOW_STYLE CV_WINDOW_AUTOSIZE  // CV_WINDOW_FREERATIO CV_WINDOW_AUTOSIZE
   _2DLeftOpenCvQtWidget = new CvWidget("2D_Image_List", WINDOW_STYLE);
   ui.tab2DHorizontalLayout->addWidget(_2DLeftOpenCvQtWidget);

   _2DRightOpenCvQtWidget = new CvWidget("3D_Image_List", WINDOW_STYLE);
   ui.tab2DHorizontalLayout->addWidget(_2DRightOpenCvQtWidget);  

   _2DPlot           = new Plot2D();
   ui.tab3DHorizontalLayout->addWidget(_2DPlot);  
  
  
  //ui.tab3DHorizontalLayout->addWidget(_2DRightOpenCvQtWidget);

  connect( ui.openImageButton, SIGNAL(clicked()),  this, SLOT(openImageButtonSlot()));
  connect( ui.drawMaskButton, SIGNAL(clicked()),  this, SLOT(drawMaskButtonSlot()));
  connect( ui.runButton, SIGNAL(clicked()),  this, SLOT(runButtonSlot()));
  
  ui.drawMaskButton->setDisabled(true);
  ui.runButton->setDisabled(true);
	_lastOpenedDir = QDir::currentPath();
	worker = NULL;
}

RegionalSegmentationUI::~RegionalSegmentationUI()
{
  delete _2DLeftOpenCvQtWidget;
  delete _2DRightOpenCvQtWidget;
  delete _2DPlot;
}

void RegionalSegmentationUI::setImage( CvWidget*& widget, Mat& image )
{
  const InputArray& _img  = image;
  Mat img = _img.getMat();
  CvMat c_img = img;    
  widget->updateImage(&c_img);
}

void RegionalSegmentationUI::openImageButtonSlot()
{ 
  //if (ui.tabWidget->currentIndex())
  //{ // 3D Tab
  //  QStringList fileList = QFileDialog::getOpenFileNames(  this, tr("Pick Multiple Image Files"),
  //                                       QDir::currentPath(),    tr("Images (*.png *.bmp *.jpg)"));
  //  if (fileList.isEmpty())
  //    return; 

  // _3dImages.clear();
  //  foreach(QString file, fileList)
  //  {
  //    Mat image = imread(file.toStdString());
  //    _3dImages.append( QPair<cv::Mat, QString>(image,file));
  //    setImage(_2DRightOpenCvQtWidget, _3dImages[0].first);
  //  }
  //  ui.runButton->setDisabled(false);
  //}
  //else
  { // 2D Tab
	QString file = QFileDialog::getOpenFileName(  this, tr("Pick an Image File"),
										 _lastOpenedDir,    tr("Images (*.png *.bmp *.jpg *.tif)"));
	
	if(file.isNull())
	  return; //No filename selected 
	_lastOpenedDir = file; 

	_2dImage.second = file; 
	_2dImage.first = imread(file.toStdString());

	setImage(_2DRightOpenCvQtWidget, _2dImage.first);
	setImage(_2DLeftOpenCvQtWidget, _2dImage.first);

	Size size = _2dImage.first.size();
	size.width *= 2;
	size.width += ui.runButton->size().width() + 30;
	size.height += ui.textBrowser->size().height() + 95;

	this->resize( QSize(size.width,size.height)); 
	
	ui.drawMaskButton->setDisabled(false);
	Size s = _2dImage.first.size();
	  maskInfo.first.setX(s.width/2);
	  maskInfo.first.setY(s.height/2);
	  maskInfo.second = min(maskInfo.first.x(), maskInfo.first.y())/2;
	
	drawMask2D();
	drawMask3D();
	ui.runButton->setDisabled(false);
  }  
  
}

void RegionalSegmentationUI::drawMaskButtonSlot()
{
 
  ui.runButton->setDisabled(true);
  ui.drawMaskButton->setDisabled(true);
  ui.openImageButton->setDisabled(true);  
  ui.textBrowser->append("The mask is a circle. Select first the center of it!");  
  _2DLeftOpenCvQtWidget->setCursor(QCursor(Qt::CrossCursor));  
  _2DLeftOpenCvQtWidget->installEventFilter(this);
  setImage(_2DLeftOpenCvQtWidget, _2dImage.first);
}

void RegionalSegmentationUI::runButtonSlot()
{
  ui.textBrowser->append("\r\n");
  ui.runButton->setDisabled(true);
  ui.drawMaskButton->setDisabled(true);
  ui.openImageButton->setDisabled(true);
  ui.spinBoxPrecision->setDisabled(true);
  ui.spinBoxScale->setDisabled(true);
  ui.spinBoxMaxIteration->setDisabled(true);
  ui.spinBoxDrawDelay->setDisabled(true);


  //if (ui.tabWidget->currentIndex())◙
  //{ // 3D Tab

  //}
  //else
  { // 2D Tab
	Mat mask = Mat::zeros(_2dImage.first.size(), CV_8U);    
	circle(mask, cv::Point(maskInfo.first.x(), maskInfo.first.y()), maskInfo.second, 1, -1);
	if( worker )
		delete worker;
	
	worker = new RegionalSegmentationWorker(this, _2dImage.first.clone(), mask, 
	  ui.spinBoxMaxIteration->value(), ui.spinBoxScale->value(), ui.spinBoxPrecision->value());
	worker->_sleepTime = ui.spinBoxDrawDelay->value();
	connect(worker, SIGNAL(finished()), this, SLOT(algorithmOver()));
	connect(worker, SIGNAL(drawTemp()), this, SLOT(drawTemp()));    
	connect(worker, SIGNAL(message(const QString&)), ui.textBrowser, SLOT(append(const QString&)));
	worker->start();
	//setImage(_2DRightOpenCvQtWidget, levelSetBSpline.drawContourAndMask(true));
  }
  
}
bool RegionalSegmentationUI::eventFilter( QObject *watched, QEvent *e )
{
  if (watched == _2DLeftOpenCvQtWidget)
  {
	switch (e->type())
	{
	case QEvent::MouseButtonPress:
	  {
		drawMaskCenterPoint(static_cast<QMouseEvent*>(e));
		break;
	  }
	case QEvent::MouseButtonRelease:
	  {
		drawMaskFinish(static_cast<QMouseEvent*>(e));
		break;
	  }
	case QEvent::MouseMove:
	  {
		if( isMaskDrawing)
		  drawCircleOnOutPut(static_cast<QMouseEvent*>(e));      
		break;
	  }
	}
  }  
  return QDialog::eventFilter(watched, e);
}


void RegionalSegmentationUI::drawMaskCenterPoint(QMouseEvent* info)
{
	  if(info->button() != Qt::LeftButton && info->button() != Qt::RightButton)
		  return;
	  ui.textBrowser->append("And now its radius!");
	  isMaskDrawing = true;
	  saveMaskTemp = maskInfo;
	  maskInfo.first = info->pos();
	  maskInfo.first.setY( maskInfo.first.y() - 28); // Adjust for the toolbar
	  const unsigned int w = _2dImage.first.size().width; 
	  if ( w < 310) maskInfo.first.setX( maskInfo.first.x() - ((310-w) >> 1)); // Adjust for the toolbar
}


void RegionalSegmentationUI::drawCircleOnOutPut( QMouseEvent* info )
{ 
  QPoint p =  info->pos(); 
  p.setY( p.y() - 28); // Adjust for the toolbar
  const unsigned int w = _2dImage.first.size().width; 
  if ( w < 310)   p.setX( p.x() - ((310-w) >> 1)); // Adjust for the toolbar

  int  x = maskInfo.first.x() - p.x(); 
  int y =  maskInfo.first.y() - p.y(); 
  maskInfo.second = sqrt((double) (x*x + y*y) );    
  drawMask2D();
}

void RegionalSegmentationUI::drawMaskFinish( QMouseEvent* info )
{
  drawCircleOnOutPut(info);
  if (!maskInfo.second)
  {
	maskInfo = saveMaskTemp;
	drawMask2D();    
  }    
  drawMask3D();
  isMaskDrawing = false;

  _2DLeftOpenCvQtWidget->removeEventFilter(this);
  _2DLeftOpenCvQtWidget->setCursor(QCursor(Qt::ArrowCursor));  
  
  ui.runButton->setDisabled(false);
  ui.drawMaskButton->setDisabled(false);
  ui.openImageButton->setDisabled(false);;
}

void RegionalSegmentationUI::drawMask2D()
{
  Mat temp = _2dImage.first.clone();
  circle(temp, cv::Point(maskInfo.first.x(), maskInfo.first.y()), maskInfo.second,
						 cv::Scalar( 0, 255, 182 ));
  setImage(_2DLeftOpenCvQtWidget, temp); 
}

void RegionalSegmentationUI::algorithmOver()
{ 
	disconnect(worker, SIGNAL(finished()), this, SLOT(algorithmOver()));
	disconnect(worker, SIGNAL(drawTemp()), this, SLOT(drawTemp()));    
	disconnect(worker, SIGNAL(message(const QString&)), ui.textBrowser, SLOT(append(const QString&)));

  delete worker;
  worker = NULL;

  ui.runButton->setDisabled(false);
  ui.drawMaskButton->setDisabled(false);
  ui.openImageButton->setDisabled(false);
  ui.spinBoxPrecision->setDisabled(false);
  ui.spinBoxScale->setDisabled(false);
  ui.spinBoxMaxIteration->setDisabled(false);
  ui.spinBoxDrawDelay->setDisabled(false);
}

void RegionalSegmentationUI::drawTemp()
{
  if(!worker->_mutex.tryLock())
	return; 

  Mat temp =  255* worker->levelSetBSpline.tempMask().clone();
  setImage(_2DRightOpenCvQtWidget, temp);
  _2DPlot->update(&worker->levelSetBSpline.LevelSetFunction().clone(), 
				   worker->levelSetBSpline.normInterValum());
  
  Mat m = _2dImage.first.clone();
  worker->levelSetBSpline.drawContour(true, m);
  setImage(_2DLeftOpenCvQtWidget, m);

  worker->_mutex.unlock();
}

void RegionalSegmentationUI::drawMask3D()
{
  Mat mask = Mat::zeros(_2dImage.first.size(), CV_8U);    
  circle(mask, cv::Point(maskInfo.first.x(), maskInfo.first.y()), maskInfo.second, 1, -1);
  Mat temp = LevelSetSegmentation::BSPlineLevelSet::maskToSignedDistanceFunction(mask);

	const double Linf =  3/ *std::max_element(temp.begin<double>(), temp.end<double>(),
										[](double i, double j) -> bool { return abs(i) < abs(j); });  
  std::for_each(temp.begin<double>(), temp.end<double>(), [&Linf](double& i) { i *= Linf;}); /*Normalize the surface.*/
  _2DPlot->update(&temp,  3);
}