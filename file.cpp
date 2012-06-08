/*
	(C) 2012 Jeff Chien
	
	Save to and load from csv files
 */

#include <fstream>
#include <iostream>
#include <string>

#include "common.h"
#include "cuda.h"
#include "file.h"

int pdla::save_to_file(pdla_result_t p, const char* file)
{
	std::ofstream resultCSV(file);
	if(!resultCSV.is_open())
		return(0);

	for(int i = 0; i < p.pos.size(); i++)
	{
		resultCSV << p.time[i] << "," << p.pos[i].x << "," << p.pos[i].y << std::endl;
	}
	resultCSV << "Time (s)," << p.elapsed << std::endl;
	resultCSV << "Time (min)," << p.elapsed / 60.0f << std::endl;
	resultCSV.close();
		
	return(1);
}

int pdla::load_from_file(pdla_result_t& p, const char* file)
{
	std::ifstream data(file);
	if(!data.is_open())
		return(0);
	
	char c;
	float time, x, y;
	std::string trash;
	while(data.good())
	{
		data >> time >> c >> x >> c >> y;
		getline(data, trash);

		p.time.push_back(time);
		p.pos.push_back(pdla::vec(x, y));
	}
	return(1);
}
