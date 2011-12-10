//*************************************************************************************************
// Echo Recognition Image Segmentation Framework using the Level-Set Technique with BSplines
// Copyright (S) 2011 Gábor Bernát
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

#include "BSPlineLevelSet.h"
#include "Filter.h"

#define  _USE_MATH_DEFINES
#include <math.h>

#include <iostream>
#include <algorithm>
#include <limits>
#include <fstream>

#include <opencv2/imgproc/imgproc.hpp>
#include <gtest/gtest.h>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
using namespace LevelSetSegmentation;

//*************************************************************************************************
// Purpose:   Default Constructor -> Initialize with default parameters
// Method:    LevelSetSegmentation::BSPlineLevelSet::BSPlineLevelSet -  public
//*************************************************************************************************
BSPlineLevelSet::BSPlineLevelSet(void)
  : _max_Iteration_Nr(100), _scale(0), _precision(0), _I(0,0, CV_64F), _Mask(0,0, CV_8U),
    _last_Iteration_Nr(0), _J(0), _runFinished(true),
    _normInterValum(3), _epsilonHD(0.5),   _alfaDivFail(3/2.0), _alfaMulOk(1.0)
{
}

//*************************************************************************************************
// Purpose:   Default deconstruction
// Method:    LevelSetSegmentation::BSPlineLevelSet::~BSPlineLevelSet -  public
//*************************************************************************************************
BSPlineLevelSet::~BSPlineLevelSet(void)
{
}

//*************************************************************************************************
// Purpose:   Get the value of the scale. Read only.
// Method:    LevelSetSegmentation::BSPlineLevelSet::Scale -  public
// Returns:   const unsigned int&
// Qualifier: const
//*************************************************************************************************
const unsigned int& BSPlineLevelSet::Scale() const
{
  return _scale;
}

//*************************************************************************************************
// Purpose:   Set/Modify the value of the scale.
// Method:    LevelSetSegmentation::BSPlineLevelSet::Scale -  public
// Returns:   unsigned int&
//*************************************************************************************************
unsigned int& BSPlineLevelSet::Scale()
{
  return _scale;
}

//*************************************************************************************************
// Purpose:   Get the value of the maximal iterations allowed for the algorithm. Read only.
// Method:    LevelSetSegmentation::BSPlineLevelSet::MaxIterationNr -  public
// Returns:   const unsigned int&
// Qualifier: const
//*************************************************************************************************
const unsigned int& BSPlineLevelSet::MaxIterationNr() const
{
  return _max_Iteration_Nr;
}

//*************************************************************************************************
// Purpose:   Set/modify the value of the maximal iterations allowed for the algorithm.
// Method:    LevelSetSegmentation::BSPlineLevelSet::MaxIterationNr -  public
// Returns:   unsigned int&
//*************************************************************************************************
unsigned int& BSPlineLevelSet::MaxIterationNr()
{
  return _max_Iteration_Nr;
}

//*************************************************************************************************
// Purpose:   Get the value we consider a valid modification of the mask (in pixel count). Read Only.
// Method:    LevelSetSegmentation::BSPlineLevelSet::Precision -  public
// Returns:   const unsigned &
// Qualifier: const
//*************************************************************************************************
const unsigned & BSPlineLevelSet::Precision() const
{
  return _precision;
}

//*************************************************************************************************
// Purpose:   Set/Modify the value we consider a valid modification of the mask (in pixel count).
// Method:    LevelSetSegmentation::BSPlineLevelSet::Precision -  public
// Returns:   unsigned int&
//*************************************************************************************************
unsigned int& BSPlineLevelSet::Precision()
{
  return _precision;
}

//*************************************************************************************************
// Purpose:   Get the Image on what we are running the segmentation algorithm. (Prefer double type)
// Method:    LevelSetSegmentation::BSPlineLevelSet::Im -  public
// Returns:   const Mat&
// Qualifier: const
//*************************************************************************************************
const Mat& BSPlineLevelSet::Im() const
{
  return _Im;
}

//*************************************************************************************************
// Purpose:   Set/Modify the Image on what we are running the segmentation algorithm.(Prefer double)
// Method:    LevelSetSegmentation::BSPlineLevelSet::Im -  public
// Returns:   Mat&
//*************************************************************************************************
Mat& BSPlineLevelSet::Im()
{
  return _Im;
}

//*************************************************************************************************
// Purpose:   Get the mask of the phi function for inner regions. Read only.
//            Prefer of from unsigned char with 1 = inner region, 0 = outer region.
//            After a segmentation this will also hold the segmentation result delimiting the inner
//            and outer regions with the upper 1,0 codding.
// Method:    LevelSetSegmentation::BSPlineLevelSet::Mask -  public
// Returns:   const Mat&
// Qualifier: const
//*************************************************************************************************
const Mat& BSPlineLevelSet::Mask() const
{
  return _Mask;
}

//*************************************************************************************************
// Purpose:   Set/Modify the mask of the phi function for inner regions.
//            Prefer of from unsigned char with 1 = inner region, 0 = outer region.
//            After a segmentation this will also hold the segmentation result delimiting the inner
//            and outer regions with the upper 1,0 codding.
// Method:    LevelSetSegmentation::BSPlineLevelSet::Mask -  public
// Returns:   Mat&
//*************************************************************************************************
Mat& BSPlineLevelSet::Mask()
{
  return _Mask;
}

//*************************************************************************************************
// Purpose:   Read the number of iterations completed at this point by the algorithm.
// Method:    LevelSetSegmentation::BSPlineLevelSet::LastIterationNr -  public
// Returns:   const unsigned int&
// Qualifier: const
//*************************************************************************************************
const unsigned int& BSPlineLevelSet::LastIterationNr() const
{
  return _last_Iteration_Nr;
}

//*************************************************************************************************
// Purpose:   Set/Modify the number of iterations completed at this point by the algorithm.
// Method:    LevelSetSegmentation::BSPlineLevelSet::LastIterationNr -  public
// Returns:   unsigned int&
// Qualifier:
//*************************************************************************************************
unsigned int& BSPlineLevelSet::LastIterationNr()
{
  return _last_Iteration_Nr;
}

//*************************************************************************************************
// Purpose:   Read the current energy value of our energy criterion.
// Method:    LevelSetSegmentation::BSPlineLevelSet::EnergyFunctionValue -  public
// Returns:   const double&
// Qualifier: const
//*************************************************************************************************
const double& BSPlineLevelSet::EnergyFunctionValue() const
{
  return _J;
}

//*************************************************************************************************
// Purpose:   Set/Modify the current energy value of our energy criterion.
// Method:    LevelSetSegmentation::BSPlineLevelSet::EnergyFunctionValue -  protected
// Returns:   double&
//*************************************************************************************************
double& BSPlineLevelSet::EnergyFunctionValue()
{
  return _J;
}

//*************************************************************************************************
// Purpose:   This parameter will balance the influence of the inner region.
// Method:    LevelSetSegmentation::BSPlineLevelSet::nuIn -  public
// Returns:   const double&
// Qualifier: const
//*************************************************************************************************
const double& BSPlineLevelSet::nuIn() const
{
  return _nuIn;
}

//*************************************************************************************************
// Purpose:   Set/Modify the inner region balancing parameter.
// Method:    LevelSetSegmentation::BSPlineLevelSet::nuIn -  protected
// Returns:   double&
//*************************************************************************************************
double& BSPlineLevelSet::nuIn()
{
  return _nuIn;
}

//*************************************************************************************************
// Purpose:   This parameter will balance the influence of the outer region.
// Method:    LevelSetSegmentation::BSPlineLevelSet::nuOut -  public
// Returns:   const double&
// Qualifier: const
//*************************************************************************************************
const double& BSPlineLevelSet::nuOut() const
{
  return _nuOut;
}

//*************************************************************************************************
// Purpose:   Set/Modify the outer region balancing parameter.
// Method:    LevelSetSegmentation::BSPlineLevelSet::nuOut -  protected
// Returns:   double&
//*************************************************************************************************
double& BSPlineLevelSet::nuOut()
{
  return _nuOut;
}

//*************************************************************************************************
// Purpose:   Initialize the values
// Method:    LevelSetSegmentation::BSPlineLevelSet::init -  protected
// Returns:   void
//*************************************************************************************************
void LevelSetSegmentation::BSPlineLevelSet::init()
{
  /*-------------------------- Input validation	----------------------------------------------*/
  _size = Im().size();
  ASSERT_EQ(_size , Mask().size()) << "Mask and image size must agree."              << endl;
  ASSERT_EQ(2    , Im().dims    ) << "Input Image must be of a 2 dimensionality."   << endl;

  /*-------------------------- Convert the input to a gray scale one and float type ------------*/
  Mat grayI, doubleI;
  if (Im().channels() != 1)
    cvtColor(Im(), grayI, CV_BGR2GRAY);
  else
    grayI = Im();

  if (CV_64F == grayI.type())
    doubleI = grayI;
  else
    grayI.convertTo(doubleI, CV_64F);

  /*--------------------------- Scale up/down the input image and the mask  -------------------*/
  int wMul = _size.width  >> Scale();   /* Divide by 2^Scale()*/
  int hMul = _size.height >> Scale();

  Size scaledSize;                    /* If we multiply back and we get the same no need to scale*/
  scaledSize.width  = ( wMul << Scale() == _size.width ) ? _size.width  : (wMul+1) << Scale();
  scaledSize.height = ( hMul << Scale() == _size.height) ? _size.height : (hMul+1) << Scale();
  /* Otherwise scale with an order higher. */
  if (scaledSize == _size)                    /* No need to expand. Just assign.        */
  {
    _I = doubleI;
    _mask = Mask().clone();
    setToZeroIfLessThanZeroUChar(_mask);
  }
  else
  {
    int i,j;								                  /* Expand at borders by holding the value*/
    double borderValue;
    double* dI, *nI;
    uchar * dM, *nM;

    _I.create(scaledSize, doubleI.depth());	 /* Create a new one and copy in the elements*/
    _mask = Mat::zeros(scaledSize, CV_8U);

    for(i = 0; i < doubleI.rows; ++i)
    {
      dI = doubleI.ptr<double>(i);
      nI =       _I.ptr<double>(i);
      dM =  Mask().ptr<uchar >(i);
      nM =    _mask.ptr<uchar >(i);
      for(j = 0; j < doubleI.cols; ++j)	     /* Assign existing elements*/
      {
        nI[j] = dI[j];
        nM[j] = (dM[j] <= 0) ? 0 : 1;
      }

      borderValue = dI[j-1];				         /* On the side hold the one at the end*/
      for( ; j < _I.cols; ++j)
        nI[j] = borderValue;
    }

    dI = _I.ptr<double>(i-1);		         /* In the bottom hold the one in the top (only for image)*/
    for(; i < _I.rows; ++i)
    {
      double* nI =       _I.ptr<double>(i);
      for(j = 0; j < _I.cols; ++j)
        nI[j] = dI[j];
    }
  }
  /* ------------------------- Filter selection ------------------------------------------------*/
  _filter = Filter::GetFilter(Scale());

  /*---------------- Initialize Phi from the mask with the Signed Distance function ------------*/
  LevelSetFunction() = maskToSignedDistanceFunction(_mask);
  /*----------------- Create the BSpline coefficients from the phi -----------------------------*/
  _BSpline = scalePhiAndCreateBSplineFromIt(LevelSetFunction(), Scale());
  /* ----------------- Initialize loop variables ----------------------------------------------*/
  LastIterationNr() = 0;
  _prevMask = _mask.clone();
  
  _count =0;
  nuIn() = 0;
  nuOut() = 0;
  EnergyFunctionValue() = numeric_limits<double>::max();
  minimizeFeatureParameters(LevelSetFunction(), _I, EnergyFunctionValue(),nuIn(), nuOut());
}

//*************************************************************************************************
// Purpose:   Start running the BSPline level set algorithm using the objects parameters you set.
// Method:    LevelSetSegmentation::BSPlineLevelSet::Run -  public
// Returns:   bool - True if the optimization finished, false otherwise (call again for a new step)
//*************************************************************************************************
bool BSPlineLevelSet::Run()
{
  if (_runFinished)
  {
    init();
    _runFinished = false;
  }
  /*-- Start and loop the minimization process until we reach the desired precision on the mask --*/
  minimize(_BSpline, LevelSetFunction(), _filter, _I);
  transform(LevelSetFunction().begin<double>(), LevelSetFunction().end<double>(), _mask.begin<uchar>(),
            [](const double& inp) -> uchar { return (inp <= 0) ? 0 : 1;});

  _count = hasReachedPrecisionDifference(_prevMask, _mask) ? ++ _count : 0;

  if( _count >= 6)                               /* After 6 successful approximations stop.     */
       _runFinished = true;
  else
  {
      ++LastIterationNr();                     /* Otherwise continue from the current one*/
      _prevMask = _mask.clone();
  }
  
  _runFinished = _runFinished || !(MaxIterationNr() - LastIterationNr());

  /*-------------------------- Reset the input sizes for the Phi and the Mask --------------------*/
  if (_runFinished)
  {
    LevelSetFunction() = LevelSetFunction()(Range(0, _size.height), Range(0, _size.width)).clone();
    Mask() = _mask(Range(0, _size.height), Range(0, _size.width)).clone();
  }  

  return _runFinished;
}

//*************************************************************************************************
// Purpose:   Create a signed distance function from the input mask binary (0 and other values) image.
// Method:    LevelSetSegmentation::BSPlineLevelSet::maskToSignedDistanceFunction -  protected
// Qualifier: const
// Returns:   cv::Mat           - OUT -> The signed distance function.
// Parameter: const Mat & mask  - IN  -> The input binary image.
//*************************************************************************************************
Mat BSPlineLevelSet::maskToSignedDistanceFunction(const Mat& mask )
{
  Mat inner, outer, offset, result;

  distanceTransform(  mask, inner, CV_DIST_L2, CV_DIST_MASK_PRECISE);
  distanceTransform(1-mask, outer, CV_DIST_L2, CV_DIST_MASK_PRECISE);

  mask.convertTo(offset, CV_32F, 1, -0.5);  //offset = mask - 0.5;
  result = outer - inner;
  result += offset;

  Mat temp;
  result.convertTo(temp, CV_64F);
  return temp;
}

//*************************************************************************************************
// Purpose:   Create the BSpline Coefficients from the input phi function values.
 //           We effectively interpolate its values, sampling the input with the scalar step.
// Method:    LevelSetSegmentation::BSPlineLevelSet::scalePhiAndCreateBSplineFromIt -  protected
// Qualifier: const
// Returns:   cv::Mat                     -    OUT -> The BSpline Coefficient Matrix.
// Parameter: Mat & phi                   - IN/OUT -> The level set function. We
// Parameter: const unsigned int & scalar - IN     -> The step we use to sample the level set.
//*************************************************************************************************
Mat BSPlineLevelSet::scalePhiAndCreateBSplineFromIt( Mat& phi, const unsigned int& scalar ) const
{
  Mat phiDownScale, BSpline;
  Size s = phi.size();
  s.width /= 1UL << scalar;
  s.height /= 1UL << scalar;

  resize(phi, phiDownScale,s, 1,1, INTER_CUBIC);           /* Down sample phi with scale.        */

  BSpline = createBSplineFromMatrix2D(phiDownScale);                /* Get its BSpline Coefficients.      */
  const double Linf = _normInterValum/ *max_element(BSpline.begin<double>(), BSpline.end<double>(),
                                       [](double i, double j) -> bool { return abs(i) < abs(j); });
  for_each(BSpline.begin<double>(), BSpline.end<double>(), /*Normalize the Coefficients.         */
                [&Linf](double& i) { i *= Linf;});

  phiDownScale = create2DInterpolationFromBSpline(BSpline);/* Convert back to the level set func. */
  resize(phiDownScale, phi, phi.size(), 1, 1, INTER_CUBIC);    /* Up Sample the found phi with scale  */
                                                           /* to form the final phi.              */
  return BSpline;
}

//*************************************************************************************************
// Purpose:   Convert a 2D Image to BSpline coefficients.
// Method:    LevelSetSegmentation::BSPlineLevelSet::createBSplineFromMatrix2D -  protected
// Qualifier: const
// Returns:   cv::Mat - OUT -> The BSpline coefficient matrix.
// Parameter: Mat & I -  IN -> The input image matrix.
//*************************************************************************************************
Mat BSPlineLevelSet::createBSplineFromMatrix2D(const Mat& I )const
{
  Mat BSpline, t; I.copyTo(BSpline);          /* We will do in place computation so we need */
  int i;                                      /* first to create a copy of the input.       */
  for ( i = 0; i < BSpline.rows; ++i)         /* We find the coefficients as 1D, first via the rows*/
    convertToBSplineCoeff(t = BSpline.row(i));

  for( i =0; i < BSpline.cols; ++i)           /* and later along the columns. We can do this cause*/
    convertToBSplineCoeff( t = BSpline.col(i)); /* the BSPline of dimension d is separable to the*/
                                                /* product of d count 1D BSplines.               */
  return BSpline;                               /* We call this orthogonal splitting.             **/
}

//*************************************************************************************************
// Purpose:   Convert a strictly 1D input vector (passed on as a matrix of 1Xn or nX1) to BSpline
//            coefficients.
// Method:    LevelSetSegmentation::BSPlineLevelSet::convertToBSplineCoeff -  protected
// Qualifier: const
// Parameter: Mat & In   - IN/OUT -> The interpolated values and the BSpline coefficients. (In place
//                                   computation used for speed and memory efficiency.
//*************************************************************************************************
void BSPlineLevelSet::convertToBSplineCoeff( Mat& In) const
{
  // The BSpline interpolates by its coefficients: s[k] = (b ☆ cP)[k] - convolution.
  // Therefore finding the coefficients from s[k] is done via defining the inverse convolution op.
  // cP[k] = b^-1  ☆ s[k]. Since b is a symmetric FIR, the b^-1 direct BSpline filter is an
  // all pole system that can be implemented efficiently using a cascade first order casual and
  // anti casual recursive filter.
  // s[k] -> [1/(1-z1*z)] ->cP+[k]-> [-z1/(1-z1*z)] -> cP-[k] =cP[k]. Here z1 is the pole of B-Spline
  // In case of a cubic BSpline: b    = (z + 4 + z^-1)/6
  //                        So   b^-1 = 6/(z + 4 + z^-1) = 6 * (1/(1-z1*z)) * (-z1/(1-z1*z))                                                                */
  // Which is equivalent to:     b^-1 = z*(1-z1)*(1-z1^-1) /( (z - z1) *(z - z1^-1));
  // The recursive algorithm looks like:
  // cP+[k] = s[k] + z1 * cP+[k-1]     -> The casual filter.  Running from left to right.
  // cP-[k] = z1 * (cP-[k+1] - cP+[k]); -> The anti casual filter. Run from right to left.
  // cP+[0] and cP-[N-1](the starting values per filter)- called casual and anti casual coefficients

  const double z1 = sqrt(3.0) - 2.0;     /* The pole of the cubic BSpline (z1) is  -2 + √3.       */
  const double gain = (1- z1) * (1-1/z1);/*(1-z1)*(1-z1^-1)                                       */
  MatIterator_<double> it = In.begin<double>(), end = In.end<double>();
  for_each(it, end,
    [&gain](double& i) { i *= gain;});  /*  z*(1-z1)*(1-z1^-1)                                    */

  *it = getInitialCausalCoefficient(In, z1);       /*Casual Coefficient Initialization. */

  double prevCoeff = *it; ++it;                    /* Starting from second item apply to each the */
  for(; it != end; ++it)                           /* casual filter.                              */
  {
    *it = *it + z1 * prevCoeff;                    /* cP+[k] = s[k] + z1 * cP[k-1] */
    prevCoeff = *it;
  }

  --it;--it; prevCoeff = *it;  ++it;             /* Anti casual coefficient initialization.       */
  *it = (z1 * prevCoeff + (*it)) * z1 /(-1+z1*z1);/*cP-(N-1)=(z1/(1-z1^2))* (c+(N-1) + z1*c+(N-2))*/

  it = In.end<double>(); end = In.begin<double>();
  --it;	prevCoeff = *it; -- it;
  for( ; it != end; --it)                        /* Apply the anti casual filter                  */
   {
    *it = z1 * (prevCoeff - (*it));              /*cP-[k] = z1 * (cP-[k+1] - cP+[k])              */
    prevCoeff = *it;
  }
  *it = z1 * (prevCoeff - (*it));
}

//*************************************************************************************************
// Purpose:   Returns the starting values for the recursive FIR filter that does a 1D signal
//            interpolation. Specifying appropriate values for this is essential in order to develop
//            a process that is reversible using the same type of boundary conditions.
// Method:    LevelSetSegmentation::BSPlineLevelSet::getInitialCausalCoefficient -  protected
// Qualifier: const
// Returns:   double
// Parameter: Mat & S
// Parameter: const double & z
//*************************************************************************************************
double BSPlineLevelSet::getInitialCausalCoefficient(Mat& S, const double& z ) const
{
  // The impulse response of the casual filter is an exponential. So we can pre-calculate sk+[0] as
  //  c+[0] = ∑s(k)*z^k where k=0,∞. This would yield an exact solution. However in practice sk is
  // enough to approximate sk with a given precision. S this case the equation becomes:
  //  c+[0] = ∑s(k)*z^k where k=0,k0. Where k0> logε/log(|z|).

  const int N = S.total();
  double zk = z;
  double zn = pow(z, N -1);
  MatIterator_<double> sk = S.begin<double>(), end = S.end<double>();

  const double e = 1e-6;
  double k0 = 2 + floor(log(e)/log(abs(z)));  // k0 = 2 + logε/log(|z|).
  if ( k0 > N)	k0 = N;

  double sum = (*sk) + zn * (*(--end));
  ++sk;
  zn *= zn;

  for(int k = 1; k < k0; ++k, ++sk)
  {
    zn /= z;
    sum += (zk + zn) * (*sk);
    zk *= z;
  }

  sum /= ( 1 - pow(z, 2*N -2));

    return sum;
}

//*************************************************************************************************
// Purpose:   Calculate the interpolation of a 2D B-Spline coefficient matrix to a 2D output.
// Method:    LevelSetSegmentation::BSPlineLevelSet::create2DInterpolationFromBSpline -  protected
// Qualifier: const
// Returns:   cv::Mat             - OUT -> The interpolated matrix.
// Parameter: const Mat & BSpline - IN  -> The BSpline coefficients.
//*************************************************************************************************
Mat BSPlineLevelSet::create2DInterpolationFromBSpline(const Mat& BSpline ) const
{
  Mat Image, t; BSpline.copyTo(Image);
  int i;
  for ( i = 0; i < Image.rows; ++i)           /* Use the separable property. First calculate      */
    convertBSplineToSignal(t = Image.row(i)); /*                            -> alongside the rows */

  for( i =0; i < Image.cols; ++i)             /*                     -> then alongside the columns*/
    convertBSplineToSignal(t = Image.col(i));

  return Image;
}

//*************************************************************************************************
// Purpose:   Interpolates a 1D BSpline coefficient vector.
// Method:    LevelSetSegmentation::BSPlineLevelSet::convertBSplineToSignal -  protected
// Qualifier: const
// Parameter: Mat & BSpline - IN/OUT -> In the BSpline coefficients and output the interpolated.
//*************************************************************************************************
void BSPlineLevelSet::convertBSplineToSignal( Mat& BSpline ) const
{
  // s[x] = c[k] ☆ b[ x- k]; =>  s[j] = ƒ(j) = ∑ c[k] b(j - k)
  // In case of a cubic BSpline: b    = (z + 4 + z^-1)/6
  // The b outside b[j-1], b[j], b[j+1] is zero so we only need to perform three multiplication
  // and two additions per point.
  const double kernelFilter[3] = {1.0/6, 4.0/6, 1.0/6};

  MatIterator_<double> it = BSpline.begin<double>(), end = BSpline.end<double>(), iL, iC;

  iL = it; ++it;
  iC = it; ++it;

  // Threat the left boundary condition. Mirror the c[1] to get c[-1].
  *iL = (*iL) * (kernelFilter[1]) + 2 * (*iC) * (kernelFilter[0]);
  for( ; it != end; ++it)
  {
    *iC = *iC * (kernelFilter[1]) + (*iL) * (kernelFilter[0])
                                  + (*it) * (kernelFilter[2]);
    iL = iC;
    iC = it;
  }
  // Threat the left boundary condition. Mirror the c[N-1] to get c[N+1].
  *iC = *iC * kernelFilter[1] + 2 * (*iL) * kernelFilter[2];
}

//*************************************************************************************************
// Purpose:   Minimize the  μin and  μin out parameters.
// Method:    LevelSetSegmentation::BSPlineLevelSet::minimizeFeatureParameters -  protected
// Qualifier: const
// Returns:   void
// Parameter: const Mat & phi  - IN  -> The level set function.
// Parameter: const Mat & I    - IN  -> The intensity image.
// Parameter: double & JN      - IN/OUT -> If we manage to minimize the current value modify c.
// Parameter: double & nuIn    -    OUT -> New nu inner value.
// Parameter: double & nuOut   -    OUT -> New nu outer value
//*************************************************************************************************
void BSPlineLevelSet::minimizeFeatureParameters( const Mat& phi, const Mat& I,
                                                double& JN, double& nuIn, double& nuOut) const
{
  //  μin  = ∫ (  H(ϕ(x)))*f(x)dx1..dxd / ∫(  H(ϕ(x)))dx1..dxd
  //  μout = ∫ (1-H(ϕ(x)))*f(x)dx1..dxd / ∫(1-H(ϕ(x)))dx1..dxd
  // Which in practice means the average intensity value inside and outside the level set
  // J(x,μin,μout) = ∫    H(ϕ(x))* (I(x) - μin )^2 dx1..dxd +
  //                 ∫ (1-H(ϕ(x))* (I(x) - μout)^2 dx1..dxd
  double uInNew, uOutNew, J_new = 0;
  double sumI = 0, sumO = 0, sumH = 0, sum1H = 0;
  Mat heavSydePhi = heaviside (phi);                                        /*     H(ϕ(x) */
  Mat oneMinHeavSydePhi = 1 - heavSydePhi;                                  /* 1 - H(ϕ(x) */

  MatConstIterator_<double> atI = I.begin<double>(), endI = I.end<double>(),
                            atH = heavSydePhi.begin<double>(), at1H = oneMinHeavSydePhi.begin<double>();

  for(; atI != endI; ++atI, ++atH, ++at1H)
  {
    sumI += (*atI) * (*atH);          /*∫(  H(ϕ(x)))*f(x)dx1..dxd */
    sumO += (*atI) * (*at1H);         /*∫(1-H(ϕ(x)))*f(x)dx1..dxd */
    sumH += (*atH);                   /*∫(  H(ϕ(x)))dx1..dxd      */
    sum1H += (*at1H);                 /*∫(1-H(ϕ(x)))dx1..dxd      */
  }

  uInNew  = sumI / sumH;      /*∫ (  H(ϕ(x)))*f(x)dx1..dxd / ∫(  H(ϕ(x)))dx1..dxd*/
  uOutNew = sumO / sum1H;     /*∫ (1-H(ϕ(x)))*f(x)dx1..dxd / ∫(1-H(ϕ(x)))dx1..dxd*/

  atI = I.begin<double>();	endI = I.end<double>();
  atH = heavSydePhi.begin<double>();	at1H = oneMinHeavSydePhi.begin<double>();

  double uI, uO;
  for(; atI != endI; ++atI, ++atH, ++at1H)
  {
    uI = (*atI) - uInNew;                           /*I(x) - μin */
    uO = (*atI) - uOutNew;                          /*I(x) - μout*/
    J_new += uI * uI * (*atH) + uO * uO * (*at1H);  /*J(x) = ∫ H(ϕ(x))* uI^2 + ∫ (1-H(ϕ(x))* uOut*/
  }
  if ( J_new < JN)   /* If we minimized c save c. Otherwise continue minimizing the coefficients*/
  {
    JN    = J_new;
    nuIn  = uInNew;
    nuOut = uOutNew;
  }
}

//*************************************************************************************************
// Purpose:   Calculate the Heaviside functions C∞ regularized evaluation for the input.
// Method:    LevelSetSegmentation::BSPlineLevelSet::heaviside  -  protected
// Qualifier: const
// Returns:   cv::Mat
// Parameter: const Mat & In
//*************************************************************************************************
Mat BSPlineLevelSet::heaviside ( const Mat& In ) const
{
  //  Hε(x) = 0.5 + arctan(x/ε)/π;
  const double epsilon = _epsilonHD;
  const double invPI = 1 / M_PI;

  Mat Out(In.size(), CV_64F);
  transform(In.begin<double>(), In.end<double>(), Out.begin<double>(),
    [&epsilon, &invPI] (double x) -> double{
      return 0.5 + invPI * atan(x/epsilon); }
  );
  return Out;
}

//*************************************************************************************************
// Purpose:   Calculate the Dirac functions C∞ regularized evaluation for the input.
// Method:    LevelSetSegmentation::BSPlineLevelSet::dirac -  protected
// Qualifier: const
// Parameter: const Mat & In    - IN  -> The evaluation.
// Returns:   cv::Mat           - OUT -> The input for whichs each element to calculate.
//*************************************************************************************************
Mat BSPlineLevelSet::dirac( const Mat& In ) const
{
  //    δε=d/dx * Hε(x) ,where  Hε(x) = 0.5 + arctan(x/ε)/π;
  // So δε= 1/(π*ε) * 1/(1+ x^2/ε^2) = 1/(π*ε)/(1+ x^2/ε^2);
  const double inv_pi_mul_epsilon = 1 / (_epsilonHD * M_PI);
  const double epsilonPow2 = _epsilonHD * _epsilonHD;

  Mat Out(In.size(), CV_64F);
  transform(In.begin<double>(), In.end<double>(), Out.begin<double>(),
    [&inv_pi_mul_epsilon, &epsilonPow2] (double x) -> double{
      return inv_pi_mul_epsilon/ (1 + x*x/epsilonPow2) ; }
  );
  return Out;
}

//*************************************************************************************************
// Purpose:   This function will modify the input BSPline coefficients so the defined energy function
//            should have minimal value.
// Method:    LevelSetSegmentation::BSPlineLevelSet::minimize -  protected
// Qualifier: const
// Parameter: double & J          IN/OUT -> The current value of the energy function
// Parameter: double & nuIn       IN/OUT -> The current value of of the inner region describing par.
// Parameter: double & nuOut      IN/OUT -> The current value of of the outer region describing par.
// Parameter: Mat & BSpline       IN/OUT -> Holds the 2D BSpline coefficients (current and optimized)
// Parameter: Mat & levelSet      IN/OUT -> The current values of the level set function.
// Parameter: const Mat & filter  IN     -> The digital filter used for the gradient calculation.
// Parameter: const Mat & I       IN     -> The image on what we are running the segmentation.
//*************************************************************************************************
void BSPlineLevelSet::minimize( Mat& BSpline,       Mat& levelSet,
                                                  const Mat& filter , const Mat& I   )
{
  // Given an arbitrary energy function dependent on some parameters. Its minimization according to
  // this classically is done by using the Euler-Lagrange or the Frechet/Gateaux derivatives.
  // A novel approach to this is to use the B-Spline formulation and perform the optimization in
  // respect to its coefficients. These derivatives may be expressed as:
  // ∂J/∂(cP[k0]) = ∫ ω(x) * βn (x/h - k0) dx1...dxd
  // ω(x) =  νin* (       H(ϕ(x))*∂gin(x,ϕ(x))/∂ϕ(x)) + gin(x,ϕ(x))*δε(ϕ(x)))
  //        +νou* ( (1 - H(ϕ(x)))*∂gou(x,ϕ(x))/∂ϕ(x)) - gou(x,ϕ(x))        ) * δε(ϕ(x))
  //        +νcP * ( || ∇ϕ(x)|| )*∂gc(x,ϕ(x))/∂ϕ(x))  - div(gc(x,ϕ(x))* ∇ϕ(x)/||∇ϕ(x)||) * δε(ϕ(x));
  // To lead to a closed form solution we use the gradient-descent method:
  //   cP(k+1) = cP(k) -  λ * ∇cJ(cP(k)) where λ is the iteration step
  //                        ∇cP is the gradient of the energy relative to the BSpline coefficients
  // In order for our level set to be bounded  (avoid step or flat gradients. Furthermore if
  // we multiply an implicit function with a non-null coefficient does not change its interface.
  // Since ϕ is represented through a set of cP[k] B-Spline coefficients multip. ϕ with ε is cP[k]*ε.
  // Therefore the level set reinitialization is => cP(k+1) = cP(k)/ ||cP(k+1)||∞
  // Now let us consider the Chen-Vese functional which aim to partitioning the image into regions
  // with piecewise-constant intensity. Here:
  // J(x,μin,μout) = ∫    H(ϕ(x))* (f(x) - μin )^2 dx1..dxd +
  //                 ∫ (1-H(ϕ(x))* (f(x) - μout)^2 dx1..dxd +
  //               ν*∫   δε(ϕ(x))*||∇ϕ(x)||       dx1..dxd
  //          ω(x) =  (f(x) - νin)^2 - (f(x) - νou)^2 * δε(ϕ(x)) + ν*div(∇ϕ(x)/||∇ϕ(x)||)* δε(ϕ(x);
  //  Furthermore we set ν to zero to give equal importance to the inner and outside region and
  //  no importance to the contour. We do this in order to minimize the calculation cost.
  // In a discrete world like ours this we can write as:
  // <∇cJ>[k] = (ωε ☆ bhn)[k] where bhn is the n degree BSpline scaled down by a factor of h
  //                 ☆ = convolution
  /*--------------------------------- Compute the energy gradient -------------------------------*/
  double iV, iU, maxValue = 0;
  Mat w = Mat::zeros(I.size(), CV_64F);/*ωε = ((I(x,y) - μin)^2 - (I(x,y) - μout(x,y))^2)* δε(ϕ(x);*/
  Mat diracLevelSet = dirac(levelSet); /*         δε(ϕ(x)                                        */
  const double* atD, *atI;
  double* atW;
  int i,j;

  for (i = 0; i < I.rows; ++i)
  {
    atI = I.ptr<double>(i);
    atD = diracLevelSet.ptr<double>(i);
    atW = w.ptr<double>(i);
    for( j = 0; j < I.cols; ++j)
    {
      iU = atI[j] - nuIn();    /* f(x) - νin*/
      iV = atI[j] - nuOut();   /* f(x) - νou*/

      atW[j] = (iU * iU - iV * iV) * atD[j]; /*ωε*/
      if (maxValue < abs(atW[j]))           /* Search for the maximal value in order to C∞ regularize.*/
        maxValue = abs(atW[j]);
    }
  }
  for_each(w.begin<double>(), w.end<double>(), [&maxValue] (double& i)
      { i /= maxValue;});              /* Normalize  ωε                                           */

  Mat gradient = ComputeEnergyGradientFromBSpline(w, filter); /* <∇cJ>[k]                        */
  /*--------------------------- Compute the gradient descent with feedback adjustment------------ */
  Mat newComputedBSpline = Mat::zeros(BSpline.size(), CV_64F);
  Mat newLevelSet;

  const unsigned int itMax = 5;
  unsigned int atIt =0;
  double differenceJ = 1, lambda = 1;
  double newComputedJ = EnergyFunctionValue(), newNuIn = nuIn(), newNuOut = nuOut(), ValueLInf;
  const double* BSplineP, *gradientP;
  double* NewBSplineP;
  while( atIt < itMax) /* Use the expectation maximization technique in order */
  {                                       /* to minimize a function with multiple parameters     */
    ++atIt;	               ValueLInf = 0; /* Minimize just one per step. Keep the rest fixed.    */

    for ( i =0 ; i < BSpline.rows; ++i)
    {
      NewBSplineP = newComputedBSpline.ptr<double>(i);
      BSplineP    = BSpline.ptr<double>(i);
      gradientP   = gradient.ptr<double>(i);
      for( j = 0; j < BSpline.cols; ++j)
      {
        NewBSplineP[j] = BSplineP[j] - lambda * gradientP[j]; /*cP(i+1) = cP(i) -  λ * ∇cJ(cP(i))*/
        if (abs(NewBSplineP[j]) > ValueLInf)                  /* Acquire L∞ of the cP(i+1)      */
          ValueLInf = abs(*NewBSplineP);
      }
    }

    ValueLInf = _normInterValum/ValueLInf;
    for_each(newComputedBSpline.begin<double>(), newComputedBSpline.end<double>(),
        [&ValueLInf] (double& inp)
            {inp *= ValueLInf;}                            /* cP(i+1) = cP(i)/ ||cP(i+1)||∞*/
    );
    newLevelSet = interpolateLevelSetFromBSpline(newComputedBSpline); /*New ϕ from BSpline'.  */

    minimizeFeatureParameters(newLevelSet, I, newComputedJ, newNuIn, newNuOut); /*Update  μin, μin,J. */

    differenceJ = newComputedJ - EnergyFunctionValue();
    lambda = (differenceJ < 0) ? lambda / _alfaDivFail : lambda * _alfaMulOk;
    if (differenceJ < 0)                                            /*If we could decrease modify. */
    {
      BSpline = newComputedBSpline;
      levelSet		= newLevelSet;
      EnergyFunctionValue() = newComputedJ;
      nuIn() = newNuIn;
      nuOut() = newNuOut;
    }
    else
      break;
  }
}

//*************************************************************************************************
// Purpose:   Calculate the gradient of the energy function.
// Method:    LevelSetSegmentation::BSPlineLevelSet::ComputeEnergyGradientFromBSpline -  protected
// Qualifier: const
// Returns:   cv::Mat               - OUT -> The gradient.
// Parameter: const Mat & feature   -  IN -> The discrete feature function
// Parameter: const Mat & filter    -  IN -> The filter used for downscaling and convolution
//*************************************************************************************************
Mat BSPlineLevelSet::ComputeEnergyGradientFromBSpline( const Mat& feature, const Mat& filter ) const
{
  // <∇cJ>[k] = (ωε ☆ bhn)[k] where bhn is the n degree BSpline scaled down by a factor of h
  //                 ☆ = convolution
  // We already have calculated ωε, now we just need to do the convolution and down scale with h
  // Since the B-Spline kernel is separable we can compute this as a series of 1D convolutions

  Size featureSize = feature.size();
  Size gradientSize(featureSize.width  >> Scale(),featureSize.height >> Scale());

  Mat temp = Mat::zeros(featureSize, CV_64F); /* Hold the result of the first 1D conv*/
  Mat gradient = Mat::zeros(gradientSize, CV_64F);                       /* set.                               */

  unsigned int i = 0;
  for(; i < (unsigned int) featureSize.width; ++i )        /* Convolute via the columns*/
  {
    GetMultiScaleConvolution(feature, filter , temp, i, false);
  }

  for( i = 0; i < (unsigned int) gradientSize.height; ++i) /* Now convolute via the rows*/
    GetMultiScaleConvolution(temp, filter, gradient, i, true);

  return gradient;
}

//*************************************************************************************************
// Purpose:   Convolute the input vector with the input vector filter and upscale the result.
// Method:    LevelSetSegmentation::BSPlineLevelSet::GetMultiScaleConvolution -  protected
// Qualifier: const
// Returns:   cv::Mat           - OUT -> The convolution and scaling of the input signal.
// Parameter: const Mat & In    -  IN -> The input signal
// Parameter: const Mat & filter-  IN -> The filter to use with a given scale.
//*************************************************************************************************
void BSPlineLevelSet::GetMultiScaleConvolution( const Mat& In, const Mat& filter, cv::Mat& result,
                                                const unsigned int at, bool isRow) const
{
  const unsigned int scale = 1UL << Scale();          /* Down scaling level = 2^scale. So for 0, 1*/
  const int N = (isRow) ?  In.size().width : In.size().height ;
  const unsigned int scaledLength = N >> Scale();     /* After scaling how many elements remain.  */
  const unsigned int Nm2 = (N << 1) - 1;
  const unsigned int filterSize = (scale << 2) - 1;   /* The filter size = scale*4 - 1            */

  Mat subImage = Mat::zeros(1, filterSize, CV_64F);

  double* filteredSection = subImage.ptr<double>(0);  /* Acquire S pointers for fast matrix access.*/
  const double* filterP   =   filter.ptr<double>(0);
  unsigned int k;
  int i;
  double sum;

  if (isRow)
  {
    const double* InP	=     In.ptr<double>(at);
    double* resultP   = result.ptr<double>(at);

    for( unsigned int n = 0; n < scaledLength; ++n)
    {
      i =  (n << Scale()) - (filterSize >> 1);      /* i = n*2^scale - filterSize/2. Get n-th chunk.*/
      for(sum = 0, k = 0; k < filterSize; ++k, ++i) /* Filter the chunk of data.                    */
      {
        filteredSection[k] = ( i >= 0 && i < N) ? InP[i  ] : /* For non existing point apply the */
          (           i >= N) ? InP[N-1]*2 - InP[Nm2 - i] :
          /*i <  0*/		      InP[0  ]*2 - InP[- i]; /*symmetry constrains.*/
      sum += filteredSection[k] * filterP[k];
      }
      resultP[n] = sum;                             /* Store the result of the convolution.         */
    }
  }
  else
  {
    const double** InP = new const double*[N]; // For the columns first acquire pointer to the items
    for( int row = 0; row < N; ++row)
      InP[row] = &In.ptr<double>(row)[at];

    for( unsigned int n = 0; n < scaledLength; ++n)
    {
      i =  (n << Scale()) - (filterSize >> 1);      /* i = n*2^scale - filterSize/2. Get n-th chunk.*/
      for(sum = 0, k = 0; k < filterSize; ++k, ++i) /* Filter the chunk of data.                    */
      {
        filteredSection[k] = ( i >= 0 && i < N) ? *InP[i] : /* For non existing point apply the */
          (           i >= N) ? *InP[N-1]*2 - *InP[Nm2 - i] :
          /*i <  0*/		       *InP[0  ]*2 - *InP[- i]; /*symmetry constrains.*/
      sum += filteredSection[k] * filterP[k];
      }
       result.ptr<double>(n)[at] = sum;
    }
    delete InP;
  }
}

//*************************************************************************************************
// Purpose:   We interpolate the level set function from an input BSpline coefficient.
//            If the BSpline is down sampled we will up sample c to create the same sized ϕ.
// Method:    LevelSetSegmentation::BSPlineLevelSet::interpolateLevelSetFromBSpline -  protected
// Qualifier: const
// Returns:   cv::Mat
// Parameter: const Mat & BSpline
//*************************************************************************************************
Mat BSPlineLevelSet::interpolateLevelSetFromBSpline( const Mat& BSpline ) const
{
  // We use the cubic BSpline whose function is:
  //    f(t) = [ 1  t t^2 t^3] * M * [sk sk+1 sk+2 sk+3]'; where Pi is the ith control points coefficient
  //           [ 1  4  1 0]
  // M = 1/6 * [−3  0  3 0]
  //           [ 3 −6  3 0]
  //           [−1  3 −3 1]
  //
  //           [ 1  t t^2 t^3] * M => cubic uniform B-Spline Blending functions

  Size bS = BSpline.size();
  Size fSize(bS.width << Scale(), bS.height << Scale());
  Mat f(fSize, CV_64F);

  const unsigned int w2 = 2 * bS.width - 1;
  const unsigned int h2 = 2 * bS.height - 1;
  const unsigned int scaleLevel = 1UL << Scale();

  Mat xW(1,4, CV_64F), yW(1,4, CV_64F);
  double* pWx = xW.ptr<double>(0);  /* Cubic uniform B-Spline Blending function along rows.       */
  double* pWy = yW.ptr<double>(0);  /* Cubic uniform B-Spline Blending function along columns.    */

  Mat c(4,4, CV_64F);

  double w;

  unsigned int u,v,k,l;
  int i,j;
  double x,y, conv, convS;
  double** cP = new double*[4];
  for( i =0; i < 4; ++i)
    cP[i] = c.ptr<double>(i);

  for( u = 0; u < (unsigned int)  fSize.height; ++u)
    for( v =0; v < (unsigned int) fSize.width ; ++v)
    {
      x = u / scaleLevel;
      y = v / scaleLevel;
      i = (int) floor(x);
      j = (int) floor(y);

      w = x - i;                        // The point in which we want to know its value along x
      pWx[3] = (1.0/6.0) * w * w * w;
      pWx[0] = (1.0/6.0) + (1.0/2.0) * w * (w-1.0) - pWx[3];
      pWx[2] = w + pWx[0] - 2.0 * pWx[3];
      pWx[1] = 1.0 - pWx[0] - pWx[2] - pWx[3];

      w = y - j;                         // The point in which we want to know its value along y
      pWy[3] = (1.0/6.0) * w * w * w;
      pWy[0] = (1.0/6.0) + (1.0/2.0) * w * (w-1.0) - pWy[3];
      pWy[2] = w + pWy[0] - 2.0 * pWy[3];
      pWy[1] = 1.0 - pWy[0] - pWy[2] - pWy[3];

      #define BS BSpline.at<double>
      --i;
      for(k = 0; k < 4; ++k, ++i)
        for(j = (int) floor(y) -1, l = 0; l <  4; ++l, ++j)
         cP[k][l] =                     /* Apply the symmetry constraints on the BSpline to get   */
          (i >= 0 && i < bS.height )   ?/* unkown coefficients. More at (1*).                     */
              (j >=0 && j <  bS.width) ?   BS(i          ,          j)                             :
              (         j >= bS.width) ? 2*BS(i          , bS.width-1) - BS(i   , w2 -j):
                                         2*BS(i          , j +1)       - BS(i   ,    -j)
         :(i >= bS.height)             ?
              (j >=0 && j <  bS.width) ? 2*BS(bS.height-1,          j) - BS(h2-i,     j):
              (         j >= bS.width) ? 2*BS(bS.height-1, bS.width-1) - BS(h2-i, w2 -j):
                                         2*BS(bS.height-1,      j + 1) - BS(h2-i,    -j)
         :
              (j >=0 && j <  bS.width) ? 2*BS(i+1        ,          j) - BS(-i  ,     j):
              (         j >= bS.width) ? 2*BS(i+1        , bS.width-1) - BS(-i  , w2 -j):
                                         2*BS(i+1        ,        j+1) - BS(-i  ,    -j);
      for(convS =0, i = 0; i < 4; ++i)
      {
        for(conv = 0, j = 0; j < 4; ++j)
          conv += pWx[j] * cP[i][j];    //The convolution:  ∑ cP[k] g(x - k)-> For the rows.
        convS  += pWy[i] * conv;        //                                  -> For the columns.
      }
      f.at<double>(u,v) = convS;
    }
    delete cP;
    return f;
    //(1*)
    // Most of the BSpline formulation in the literature treats BSplines with infinite lengths.
    // On the other hand the resources of our system is finite. However, if we can predict the
    // in some way the coefficients of the BSpline (extrapolate) we can use the infinite formulation.
    // One may do this in multiple ways and there is no perfect solution. After all any of them are
    // just a wild guess, some of them more informed than others. Here we use an extrapolation
    // scheme that mirrors the coefficients around their extremities ( namely 0 and N-1). This way
    // we assure that any required coefficient outside the known region will be definite, we do not
    // need to invent special values (less computational cost) and furthermore we do not introduce
    // discontinuities atL the extremities.
}

//*************************************************************************************************
// Purpose:   Compares the two input masks to see if they are approximately the same.
//            Using the Precision() function to get desired level.
// Method:    LevelSetSegmentation::BSPlineLevelSet::hasReachedPrecisionDifference -  protected
// Qualifier: const
// Returns:   bool
// Parameter: const Mat & curMask
// Parameter: const Mat & newMask
//*************************************************************************************************
bool BSPlineLevelSet::hasReachedPrecisionDifference( const Mat& curMask, const Mat& newMask)
{
  MatConstIterator_<uchar> curMI = curMask.begin<uchar>(), curMEnd =  curMask.end<uchar>(),
                                    newMI = newMask.begin<uchar>();
  _nrOfDifferentPixels  = 0;

  for( ; curMI != curMEnd; ++curMI, ++newMI)  /* Count the number of differing pixels.*/
    _nrOfDifferentPixels +=  (*curMI != *newMI);

  return _nrOfDifferentPixels < Precision();
}

//*************************************************************************************************
// Purpose:   Set to one the values of a matrix if I(x) >= 0. Equals with operator >=.
// Method:    LevelSetSegmentation::BSPlineLevelSet::setToZeroIfLessThanZeroUChar -  protected
// Qualifier: const
// Parameter: cv::Mat & inp - IN/OUT -> The Matrix in discussion.
//*************************************************************************************************
void LevelSetSegmentation::BSPlineLevelSet::setToZeroIfLessThanZeroUChar( cv::Mat& inp ) const
{
  for_each(inp.begin<uchar>(), inp.end<uchar>(), [](uchar& inp)
                    {
                      inp = ( inp<=0) ? 0 : 1;
         });
}

//*************************************************************************************************
// Purpose:   Holds the last used level-set function.Modify.
// Method:    LevelSetSegmentation::BSPlineLevelSet::LevelSetFunction -  public
// Returns:   cv::Mat&
//*************************************************************************************************
cv::Mat& LevelSetSegmentation::BSPlineLevelSet::LevelSetFunction()
{
  return _LevelSetFunction;
}

//*************************************************************************************************
// Purpose:   Holds the last used level-set function. Read-only.
// Method:    LevelSetSegmentation::BSPlineLevelSet::LevelSetFunction -  public
// Qualifier: const
// Returns:   const cv::Mat&
//*************************************************************************************************
const cv::Mat& LevelSetSegmentation::BSPlineLevelSet::LevelSetFunction() const
{
  return _LevelSetFunction;
}

//*************************************************************************************************
// Purpose:   Draws the contour of a segmentation in a given window and waits for the given time.
// Method:    LevelSetSegmentation::BSPlineLevelSet::drawContourAndMask -  public
// Qualifier: const
//*************************************************************************************************
Mat LevelSetSegmentation::BSPlineLevelSet::drawContourAndMask( bool inverse /*=false*/) const
{
  Mat temp = (inverse) ? Mask() == 0 : Mask().clone();

  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  findContours( temp, contours, hierarchy,  CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

  Mat draw;
  //draw = 255* Mat::zeros(Mask().size(), CV_8UC3);
  cvtColor(255*Mask(), draw, CV_GRAY2BGR);

  for(int idx = 0; idx >= 0; idx = hierarchy[idx][0] )// draw each connected component with its own
  {                                       //random color, iterate through all the top-level contours
    Scalar color( rand()&255, rand()&255, rand()&255 );
    /*Scalar color( 0, 0, 255 );*/
    drawContours( draw, contours, idx, color, 1 , CV_AA, hierarchy );
  }
  return draw;
}
//*************************************************************************************************
// Purpose:   Draws the contour of a segmentation in a given window and waits for the given time.
// Method:    LevelSetSegmentation::BSPlineLevelSet::drawContourAndMask -  public
// Qualifier: const
//*************************************************************************************************
Mat LevelSetSegmentation::BSPlineLevelSet::drawContour( bool inverse /*=false*/, Mat& draw) const
{
  Mat temp = (inverse) ? _mask == 0 : _mask.clone();

  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  findContours( temp, contours, hierarchy,  CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

  for(int idx = 0; idx >= 0; idx = hierarchy[idx][0] )// draw each connected component with its own
  {                                       //random color, iterate through all the top-level contours
    /*Scalar color( rand()&255, rand()&255, rand()&255 );*/
    Scalar color( 0, 255, 182 );
    drawContours( draw, contours, idx, color, 1 , CV_AA, hierarchy );
  }
  return draw;
}


const cv::Mat& LevelSetSegmentation::BSPlineLevelSet::tempMask() const
{
  return _mask;
}

const unsigned int LevelSetSegmentation::BSPlineLevelSet::NumberOfDiferentPixelsInLastIt() const
{
  return _nrOfDifferentPixels;
}

const double LevelSetSegmentation::BSPlineLevelSet::normInterValum() const
{
  return _normInterValum;
}


