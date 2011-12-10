//*************************************************************************************************
// Echo Recognition Image Segmentation Framework using the Level-Set Technique with BSplines
// Copyright (C) 2011 Gábor Bernát
// Created at: [Sapientia - Hungarian University of Transylvania]
//
// This program is free software// you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation// either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY// without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program// if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
//*************************************************************************************************
#include "PreciseTimer.h"
#include "BSPlineLevelSet.h"
#include "gtest/gtest.h"

#include <string>
#include <iostream>
#include <sstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace LevelSetSegmentation;



#define VERBOSE_RUN_DEBUG
#define SHOW_WINDOW
#define WAIT_TIME 100 


#define ECHO_IMAGE_LOCATION "..\\Resource\\Image\\"
#define ECHO_VIDEO_LOCATION "..\\Resource\\Video\\"

#include <string>

const std::string ProcessingOutput("Image Processing Output");
const std::string OriginalImageWindow("Original Image");

static std::string getWorkDir()
{
#define MAXPATHLEN 256
  char temp[MAXPATHLEN];
  _getcwd(temp, MAXPATHLEN);
  return std::string(temp).append("\\");
}

static const std::string WorkingDirectory = getWorkDir();


int main( int argc, char* argv[])
{
//--------------------------------------------------------------------- Initialization Tasks -------
  PreciseTimer timer;
  timer.SupportsHighResCounter();

#ifdef SHOW_WINDOW
  namedWindow(ProcessingOutput, CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
  namedWindow(OriginalImageWindow, CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
#endif
//---------------------------- ---------------	Open the Image File and get its traits -------------
#define DIR  "kezek\\"
#if 0
#define NAME "Spirale_inbalanced_small.png"//"Airplane.jpg"
#else
  #define NAME "111191_UV_1_small_300.jpg"
#endif

  string imageFileName(ECHO_IMAGE_LOCATION DIR NAME );
  Mat input = imread(imageFileName);

  ASSERT_TRUE(input.data != NULL) << "Image File Not Found => "		//threat failed to open
                     << WorkingDirectory << imageFileName;

//------------------- Move the Windows so we can easily see them -----------------------------------
  cvMoveWindow(ProcessingOutput.c_str(), input.size().width+30, 367);
  cvMoveWindow(OriginalImageWindow.c_str(), 0, 367);

//---------------------------------------------------------------- Process all the frames ----------
  Mat temp;
  cvtColor(input, temp, CV_BGR2GRAY,1);

  BSPlineLevelSet levelSetBSpline;
  levelSetBSpline.Im() = temp;

  Mat mask = Mat::zeros(input.size(), CV_8U);
  Point maskCenter( temp.size().width/2, temp.size().height/2);
  circle(mask, maskCenter , min(maskCenter.x, maskCenter.y) /2, 1, -1);

  levelSetBSpline.Mask() = mask;
  levelSetBSpline.MaxIterationNr() = 100;

  levelSetBSpline.Scale() = 0 ;
  levelSetBSpline.Precision() = levelSetBSpline.Im().total() / 10;

#ifdef VERBOSE_RUN_DEBUG  
  cout << "Targeted precision: " << levelSetBSpline.Precision()<< "." << endl;
  imshow(ProcessingOutput, mask*255);
  imshow(OriginalImageWindow, levelSetBSpline.Im());
  waitKey(WAIT_TIME);
#endif 
  double energy; 
  timer.StartTimer();
  while(!levelSetBSpline.Run())
  {
    #ifdef VERBOSE_RUN_DEBUG  
      cout << "At iteration " << levelSetBSpline.LastIterationNr() 
           << " with energy: " << levelSetBSpline.EnergyFunctionValue()  << "." << endl;
      imshow(ProcessingOutput,255* levelSetBSpline.tempMask());
      waitKey(WAIT_TIME);
    #endif
    
  }
  cout << endl << "Time:            " << ( (double)timer.StopTimer() / 1000000) << " s" << endl 
               << "Iteration Count: " << levelSetBSpline.LastIterationNr()      << " #" << endl;

#ifdef SHOW_WINDOW
  temp = levelSetBSpline.drawContourAndMask(true);
  imshow(ProcessingOutput, temp);
  imshow(OriginalImageWindow, levelSetBSpline.Im());
#endif
  /*stringstream ss;
  ss << "Echo Resource\\Result\\" DIR "ResultScale" << levelSetBSpline.Scale() << NAME;
  imwrite(ss.str(), temp);*/

  cout << "Finished :D." << endl;

#ifdef SHOW_WINDOW 
  waitKey();
#endif
  return 0;
}