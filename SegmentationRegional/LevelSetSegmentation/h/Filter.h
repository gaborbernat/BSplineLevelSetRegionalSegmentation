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
#ifndef __FILTER_H__
#define __FILTER_H__

#include "General.h"
#include <opencv2/core/core.hpp>

namespace LevelSetSegmentation
{
	class DECLSPEC Filter
	{
	public:
		static cv::Mat GetFilter(const unsigned int at);
	private:
	 static double const m0[3];
	 const static double m1[7];
	 const static double m2[15];
	 const static double m3[31];
	const static double m4[63];
	};
}

#endif