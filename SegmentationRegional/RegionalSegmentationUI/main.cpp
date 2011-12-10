#include "regionalsegmentationui.h"
#include <QtGui/QApplication>

int main(int argc, char **argv)
{
  QApplication a(argc, argv);
  RegionalSegmentationUI w;
  w.show();  
  return a.exec();
}