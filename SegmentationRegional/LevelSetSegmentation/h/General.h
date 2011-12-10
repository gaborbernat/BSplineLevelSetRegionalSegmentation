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
#ifndef __GENERAL_H__
#define __GENERAL_H__

#pragma warning( disable: 4251 ) // For the cv::Mat "dll-interface to be used by clients of class"

#ifndef ECHO_RECOGNITION_EXPORTS
#define DECLSPEC __declspec(dllimport)
#else
#define DECLSPEC __declspec(dllexport)
#endif

#endif
