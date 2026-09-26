// Copyright (c) 2026 Hans-Kristian Arntzen and PyroWave contributors
// SPDX-License-Identifier: MIT
#pragma once

// The container written by encode.cpp and read by decode.cpp in the top level
// directory, so the WebGPU tools and the Vulkan ones can read each other's files:
//   "PYROWAVE", int32 params[8], then per frame a u32 size and one packet.
// params: width, height, YUV4MPEGFile::Format, chroma (0 = 420, 1 = 444),
//         full range, frame rate numerator, frame rate denominator, 0.

#include "yuv4mpeg.hpp"
#include <algorithm>
#include <chrono>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

namespace PyroWaveFile
{
struct Header
{
	int32_t width = 0;
	int32_t height = 0;
	int32_t format = 0;
	int32_t chroma = 0;
	int32_t full_range = 0;
	int32_t frame_rate_num = 0;
	int32_t frame_rate_den = 0;
	int32_t reserved = 0;
};
static_assert(sizeof(Header) == 8 * sizeof(int32_t), "Header size mismatch.");

inline bool write_header(FILE *file, const Header &header)
{
	return fwrite("PYROWAVE", 1, 8, file) == 8 && fwrite(&header, sizeof(header), 1, file) == 1;
}

inline bool read_header(FILE *file, Header &header)
{
	char magic[9] = {};
	return fread(magic, 1, 8, file) == 8 && strcmp(magic, "PYROWAVE") == 0 &&
	       fread(&header, sizeof(header), 1, file) == 1;
}

inline bool write_packet(FILE *file, const void *data, uint32_t size)
{
	return fwrite(&size, sizeof(size), 1, file) == 1 && fwrite(data, 1, size, file) == size;
}

inline bool read_packet(FILE *file, std::vector<uint8_t> &data)
{
	uint32_t size;
	if (fread(&size, sizeof(size), 1, file) != 1)
		return false;
	data.resize(size);
	return fread(data.data(), 1, size, file) == size;
}

inline const char *format_to_y4m(YUV4MPEGFile::Format fmt)
{
	switch (fmt)
	{
	case YUV4MPEGFile::Format::YUV420P: return "C420";
	case YUV4MPEGFile::Format::YUV420P16: return "C420p16";
	case YUV4MPEGFile::Format::YUV444P: return "C444";
	case YUV4MPEGFile::Format::YUV444P16: return "C444p16";
	default: return "???";
	}
}

inline std::string y4m_params(const Header &header)
{
	char params[1024];
	snprintf(params, sizeof(params), "YUV4MPEG2 W%d H%d F%d:%d Ip A1:1 XCOLORRANGE=%s %s\n",
	         header.width, header.height, header.frame_rate_num, header.frame_rate_den,
	         header.full_range ? "FULL" : "LIMITED", format_to_y4m(YUV4MPEGFile::Format(header.format)));
	return params;
}

struct Stopwatch
{
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	double elapsed_ms() const
	{
		return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
	}
};

// Mean and a few order statistics of a set of per frame times.
struct TimeStats
{
	std::vector<double> samples;
	void add(double ms) { samples.push_back(ms); }
	void print(const char *label) const
	{
		if (samples.empty())
			return;
		std::vector<double> sorted = samples;
		std::sort(sorted.begin(), sorted.end());
		double sum = 0.0;
		for (double s : sorted)
			sum += s;
		printf("  %-28s mean %8.3f ms  median %8.3f ms  p95 %8.3f ms\n", label, sum / double(sorted.size()),
		       sorted[sorted.size() / 2], sorted[std::min(sorted.size() - 1, sorted.size() * 95 / 100)]);
	}
};
}
