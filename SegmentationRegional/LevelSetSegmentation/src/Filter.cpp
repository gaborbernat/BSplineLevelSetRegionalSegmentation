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
#include "Filter.h"

using namespace cv;
using namespace LevelSetSegmentation;

 double const Filter::m0[3] =  { 0.1667, 0.6667,0.1667 };
 double const Filter::m1[7] = { 0.0208, 0.1667, 0.4792, 0.6667, 0.4792, 0.1667, 0.0208};
 double const Filter::m2[15] = {0.0026, 0.0208, 0.0703, 0.1667, 0.3151, 0.4792, 0.6120, 0.6667,
								0.6120, 0.4792, 0.3151, 0.1667, 0.0703, 0.0208, 0.0026};
 double const Filter::m3[31] = {3.2552e-004, 0.0026, 0.0088, 0.0208, 0.0407, 0.0703, 0.1117,
								 0.1667, 0.2360, 0.3151, 0.3981, 0.4792, 0.5524, 0.6120, 0.6520,
								 0.6667, 0.6520, 0.6120, 0.5524, 0.4792, 0.3981, 0.3151, 0.2360,
								 0.1667, 0.1117, 0.0703, 0.0407, 0.0208, 0.0088, 0.0026, 3.2552e-004};
 double const Filter::m4[63] = {4.0690e-005, 3.2552e-004, 0.0011, 0.0026, 0.0051, 0.0088,
								 0.0140, 0.0208, 0.0297, 0.0407, 0.0542, 0.0703, 0.0894, 0.1117,
								 0.1373, 0.1667, 0.1997, 0.2360, 0.2747, 0.3151, 0.3565, 0.3981,
								 0.4392, 0.4792, 0.5171, 0.5524, 0.5843, 0.6120, 0.6348, 0.6520,
								 0.6629, 0.6667, 0.6629, 0.6520, 0.6348, 0.6120, 0.5843, 0.5524,
								 0.5171, 0.4792, 0.4392, 0.3981, 0.3565, 0.3151, 0.2747, 0.2360,
								 0.1997, 0.1667, 0.1373, 0.1117, 0.0894, 0.0703, 0.0542, 0.0407,
								 0.0297, 0.0208, 0.0140, 0.0088, 0.0051, 0.0026, 0.0011,	3.2552e-004,
								 4.0690e-005};

Mat Filter::GetFilter( const unsigned int at )
{
	Mat filter;
	switch (at)
	{
	case 0: filter = Mat(1,3, CV_64F,(void*) Filter::m0,sizeof(double)); break;
	case 1: filter = Mat(1,7, CV_64F,(void*) Filter::m1,sizeof(double)); break;
	case 2: filter = Mat(1,15, CV_64F,(void*) Filter::m2,sizeof(double)); break;
	case 3: filter = Mat(1,31, CV_64F,(void*) Filter::m3,sizeof(double)); break;
	case 4: filter = Mat(1,63, CV_64F,(void*) Filter::m4,sizeof(double)); break;
	default:
		filter = Mat::zeros(1,1, CV_64F);
		break;
	}
	return filter;
}