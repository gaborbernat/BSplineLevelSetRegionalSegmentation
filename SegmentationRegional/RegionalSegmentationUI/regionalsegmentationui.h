#ifndef REGIONALSEGMENTATIONUI_H
#define REGIONALSEGMENTATIONUI_H

#include <QtGui/QDialog>
#include "ui_regionalsegmentationui.h"
#include <opencv2/core/core.hpp>
#include <qwt3d_surfaceplot.h>
#include <qwt3d_function.h>

class CvWidget;
class RegionalSegmentationWorker; 
namespace LevelSetSegmentation
{
class BSPlineLevelSet;
};
namespace cv 
{
  class Mat;
};

class LevelSetFunctionSurface : public Qwt3D::Function
{
public:
  LevelSetFunctionSurface(Qwt3D::SurfacePlot* pw);   
  double operator()(double x, double y);
  const cv::Mat* _pM; 
  const double scaleDrawFactor;
};

class Plot2D : public Qwt3D::SurfacePlot
{
public:  Plot2D();
         void update(const cv::Mat* m, const double n );
private:
         LevelSetFunctionSurface _levelSetFunction;
         
};

class RegionalSegmentationUI : public QDialog
{
    Q_OBJECT

public:
    RegionalSegmentationUI(QWidget *parent = 0, Qt::WFlags flags = 0);
    ~RegionalSegmentationUI();
    bool eventFilter(QObject *watched, QEvent *e);
private:
  void setImage( CvWidget*& widget, cv::Mat& image );
  void drawMaskCenterPoint(QMouseEvent* info);
  void drawCircleOnOutPut(QMouseEvent* info);
  void drawMaskFinish(QMouseEvent* info);
  void drawMask2D(); 
  void drawMask3D(); 
private slots:
  void openImageButtonSlot();
  void drawMaskButtonSlot();
  void runButtonSlot();
  void algorithmOver(); 
  void drawTemp();
  
  
private:    
    Ui::RegionalSegmentationUIClass ui;
    CvWidget* _2DLeftOpenCvQtWidget;
    CvWidget* _2DRightOpenCvQtWidget;    
    QPair<cv::Mat, QString> _2dImage; 
    QList< QPair<cv::Mat, QString> > _3dImages;
    QPair<QPoint,int> maskInfo; 
    QPair<QPoint,int> saveMaskTemp; 
    bool isMaskDrawing; 
    RegionalSegmentationWorker* worker;     
    Plot2D* _2DPlot;
    QString _lastOpenedDir;
};


#endif // REGIONALSEGMENTATIONUI_H
