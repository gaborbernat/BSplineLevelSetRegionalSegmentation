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
#ifndef __BSPLINE_LEVEL_SET_H__
#define __BSPLINE_LEVEL_SET_H__
#pragma once

#include "General.h"
#include <opencv2/core/core.hpp>


namespace LevelSetSegmentation
{
  class DECLSPEC BSPlineLevelSet
  {
  public:
    BSPlineLevelSet(void);
    virtual ~BSPlineLevelSet(void);

    bool Run(); 
    cv::Mat drawContourAndMask( bool inverse = false) const;
    cv::Mat static maskToSignedDistanceFunction(const  cv::Mat& mask);
    cv::Mat drawContour( bool inverse /*=false*/,cv::Mat& draw) const;
    
    const unsigned int& Scale() const;
    unsigned int& Scale();
    const unsigned int& MaxIterationNr() const;
    unsigned int& MaxIterationNr();
    const unsigned int&  Precision() const;
    unsigned int&  Precision();
    const cv::Mat& Im() const;
    cv::Mat& Im();
    const cv::Mat& Mask() const;
    cv::Mat& LevelSetFunction();
    const cv::Mat& LevelSetFunction() const;
    cv::Mat& Mask();
    const unsigned int& LastIterationNr() const;
    unsigned int& LastIterationNr();
    const double& EnergyFunctionValue() const;
    const double& nuIn() const;
    const double& nuOut() const;
    double& nuIn();
    double& nuOut();
    double& EnergyFunctionValue();
    const cv::Mat& tempMask() const;
    const unsigned int NumberOfDiferentPixelsInLastIt() const; 
    const double normInterValum() const; 
  protected:
    void init();
    __inline void setToZeroIfLessThanZeroUChar(cv::Mat& inp) const;
    
    cv::Mat heaviside (const cv::Mat& In) const;
    cv::Mat dirac(const cv::Mat& In) const;    
    cv::Mat scalePhiAndCreateBSplineFromIt(cv::Mat& phi, const unsigned int& scalar ) const;
    double	getInitialCausalCoefficient(cv::Mat& In, const double& z) const;
    void    convertToBSplineCoeff(cv::Mat& In)const;
    cv::Mat createBSplineFromMatrix2D(const cv::Mat& I) const;
    void    convertBSplineToSignal( cv::Mat& BSpline) const;
    cv::Mat create2DInterpolationFromBSpline(const cv::Mat& BSpline) const;
    void minimizeFeatureParameters( const cv::Mat& phi, const cv::Mat& I,
                                       double& JN, double& nuIn, double& nuOut) const;
    void minimize(       cv::Mat& BSpline,       cv::Mat& phi,
                                      const cv::Mat& filter,  const cv::Mat& I);
    cv::Mat ComputeEnergyGradientFromBSpline(const cv::Mat& feature, const cv::Mat& filter ) const;
    void GetMultiScaleConvolution( const cv::Mat& In, const cv::Mat& filter,
                                      cv::Mat& result, const unsigned int at, bool isRow) const;
    cv::Mat interpolateLevelSetFromBSpline(const cv::Mat& BSPline) const;
    bool hasReachedPrecisionDifference(const cv::Mat& curMask, const cv::Mat& newMask);
    
  protected:
    unsigned int _scale;
    unsigned int _max_Iteration_Nr;
    unsigned int _last_Iteration_Nr;
    unsigned int _precision;
    cv::Mat _Im;
    cv::Mat _Mask;
    cv::Mat _Segmentation;
    cv::Mat _LevelSetFunction;
    double	_J;
    double _nuIn;
    double _nuOut;
    bool _runFinished;
   cv::Mat _BSpline;
   cv::Mat _filter;
   cv::Mat _prevMask;
   cv::Mat _I;
   cv::Mat _mask;
   cv::Size _size;
   unsigned int _count;
   unsigned int _nrOfDifferentPixels; 
    const double _normInterValum;
    const double _epsilonHD;
    const double _alfaMulOk;
    const double _alfaDivFail;
  };
}
#endif