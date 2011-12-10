#ifndef __REGIONAL_SEGMENTATION_WORKER_H__
#define __REGIONAL_SEGMENTATION_WORKER_H__

#include <qthread.h>
#include <BSPlineLevelSet.h>
#include <QPair>
#include <QMutex>

namespace cv
{
  class Mat;
};

class RegionalSegmentationWorker : public QThread
{
  Q_OBJECT
public:
  RegionalSegmentationWorker(QObject* parent, cv::Mat& image, cv::Mat& mask, 
                  const unsigned int maxIt, const unsigned int scale, const unsigned int precision);
  virtual ~RegionalSegmentationWorker();
  void run();
signals:
  void message(const QString& message);
  void drawTemp();
public:
  LevelSetSegmentation::BSPlineLevelSet levelSetBSpline;
  QMutex _mutex; 
  unsigned int _sleepTime; 
};
#endif