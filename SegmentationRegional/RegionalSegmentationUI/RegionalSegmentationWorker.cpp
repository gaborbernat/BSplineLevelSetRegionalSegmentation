#include <RegionalSegmentationWorker.h>
#include <opencv2/core/core.hpp>
#include <PreciseTimer.h>
#include <qtextstream.h>

using namespace LevelSetSegmentation;
using namespace cv;

RegionalSegmentationWorker::RegionalSegmentationWorker( QObject* parent, Mat& image, Mat& mask, 
                                          const unsigned int maxIt, const unsigned int scale, 
                                          const unsigned int precision) : QThread(parent)
{
    levelSetBSpline.Im() = image;
    levelSetBSpline.Mask() = mask;
    levelSetBSpline.MaxIterationNr() = maxIt;
    levelSetBSpline.Scale() = scale ;
    levelSetBSpline.Precision() = precision;
    _sleepTime = 20;
}

RegionalSegmentationWorker::~RegionalSegmentationWorker()
{
}

void RegionalSegmentationWorker::run()
{
  PreciseTimer timer; 
  timer.StartTimer();
 
  QString x;   
  QTextStream(&x) << "Targeted precision: " << levelSetBSpline.Precision()<< " out of "
                  << levelSetBSpline.Im().total() << " pixels in the image.";
  emit message(x); x.clear();
  emit drawTemp();
  

  _mutex.lock();
      while(!levelSetBSpline.Run())
      {
        QTextStream(&x) << "At iteration " << levelSetBSpline.LastIterationNr() 
                        << " with energy: " << levelSetBSpline.EnergyFunctionValue()  << "."
                        << " Current precision: " << levelSetBSpline.NumberOfDiferentPixelsInLastIt() 
                        << ".";
        emit message(x);x.clear();
        Mat m = levelSetBSpline.Mask().clone();  
        Mat l = levelSetBSpline.LevelSetFunction().clone();    
        _mutex.unlock();
        emit drawTemp();
        Sleep(_sleepTime);  // to assure we do not starve the GUI thread (give chance to take over signal).
        _mutex.lock();
      }
  _mutex.unlock();

  // after
  QTextStream(&x) << "Time:                     " << ( (double)timer.StopTimer() / 1000000) << " s" << endl 
                  << "Iteration Count: " << levelSetBSpline.LastIterationNr()      << "          #";
  emit message(x);x.clear();
  emit drawTemp();
}
