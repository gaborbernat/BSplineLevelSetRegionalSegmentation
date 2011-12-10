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
using namespace std;

bool PreciseTimer::sm_bInit = false;
bool PreciseTimer::sm_bPerformanceCounter;
__int64 PreciseTimer::sm_i64Freq;

//CONSTRUCTOR
PreciseTimer::PreciseTimer() : m_i64Start(0), m_i64Elapsed(0), m_bRunning(false)
{
	//Only if not already initialized
	if(false == sm_bInit)
	{
		//Initializing some static variables dependent on the system just once
		LARGE_INTEGER liFreq;
		if(TRUE == QueryPerformanceFrequency(&liFreq))
		{
			//Only if the system is supporting High Performance
			sm_i64Freq = ((__int64)liFreq.HighPart << 32) + (__int64)liFreq.LowPart;
			sm_bPerformanceCounter = true;
		}
		else
			sm_bPerformanceCounter = false;
		sm_bInit = true;
	}
}

void PreciseTimer::StartTimer()
{
	if(true == sm_bPerformanceCounter)
	{
		QueryPerformanceCounter(&m_liCount);
		m_i64Start = ((__int64)m_liCount.HighPart << 32) + (__int64)m_liCount.LowPart;
		//Transform in microseconds
		(m_i64Start *= 1000000) /= sm_i64Freq;
	}
	else
		//Transform milliseconds to microseconds
		m_i64Start = (__int64)GetTickCount() * 1000;
	m_bRunning = true;
}

__int64 PreciseTimer::StopTimer()
{
	UpdateElapsed();
	m_bRunning = false;
	return m_i64Elapsed;
}

__int64 PreciseTimer::GetTime()
{
	if(true == m_bRunning)
		UpdateElapsed();
	return m_i64Elapsed;
}

string PreciseTimer::Int64ToString(__int64 const& ri64, int iRadix/*=10*/)
{
	bool bNeg = (ri64 < 0);
	__int64 i64 = ri64;
	string ostrRes;
	bool bSpecial = false;
	if(true == bNeg)
	{
		i64 = -i64;
		if(i64 < 0)
			// Special case number -9223372036854775808 or

			// 0x8000000000000000

			bSpecial = true;
		ostrRes.append(1, '-');
	}
	int iR;
	do
	{
		iR = i64 % iRadix;
		if(true == bSpecial)
			iR = -iR;
		if(iR < 10)
			ostrRes.append(1, '0' + iR);
		else
			ostrRes.append(1, 'A' + iR - 10);
		i64 /= iRadix;
	}
	while(i64 != 0);
	//Reverse the string

	string::iterator it = ostrRes.begin();
	if(bNeg)
		it++;
	reverse(it, ostrRes.end());
	return ostrRes;
}