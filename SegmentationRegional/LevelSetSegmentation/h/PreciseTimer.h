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

#ifndef _PRECISETIMER_H_
#define _PRECISETIMER_H_

#include "General.h"
#include <Windows.h>
#include <string>

//More precise Timer for measuring time intervals in microseconds.
//The performance of this Timer is dependent on the performance of the system.
class DECLSPEC PreciseTimer
{
public:
	//CONSTRUCTOR
	PreciseTimer();

	bool SupportsHighResCounter();
	void StartTimer();
	static std::string Int64ToString(__int64 const& ri64, int iRadix =10);
	__int64 StopTimer();
	__int64 GetTime();

private:
	//Auxiliary Function
	void UpdateElapsed();

	//Member variables
	bool m_bRunning;
	__int64 m_i64Start;
	__int64 m_i64Elapsed;

	//Some auxiliary variables
	__int64 m_i64Counts;
	LARGE_INTEGER m_liCount;

	//Static Variables
	static bool sm_bInit;
	static bool sm_bPerformanceCounter;
	static __int64 sm_i64Freq;
};

inline bool PreciseTimer::SupportsHighResCounter()
{
	return sm_bPerformanceCounter;
}

//Auxiliary Function
inline void PreciseTimer::UpdateElapsed()
{
	if(true == sm_bPerformanceCounter)
	{
		QueryPerformanceCounter(&m_liCount);
		m_i64Counts = ((__int64)m_liCount.HighPart << 32) + (__int64)m_liCount.LowPart;
		//Transform in microseconds
		(m_i64Counts *= 1000000) /= sm_i64Freq;
	}
	else
		//Transform milliseconds to microseconds
		m_i64Counts = (__int64)GetTickCount() * 1000;
	if(m_i64Counts > m_i64Start)
		m_i64Elapsed = m_i64Counts - m_i64Start;
	else
		//Eliminate possible number overflow (0x7fffffffffffffff is the maximal __int64 positive number)
		m_i64Elapsed = (0x7fffffffffffffff - m_i64Start) + m_i64Counts;
}

#endif // _PRECISETIMER_H_