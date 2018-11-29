#pragma once
#include "pch.h"
#include <iostream>
#include <math.h>
#include <string>
#include <fstream>
#include <cstdlib>
#include <random>
#include <stdio.h>
#include <limits>
#include <boost/random.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <dlib/matrix.h>
#include <dlib/optimization.h>
#include <dlib/global_optimization.h>

using namespace std;
using namespace dlib;
struct CIR_params { matrix<double> A, B; double a, b; };
static matrix<double, 0, 1>  R(639), R0(638), R1(638), tau(639);
static matrix<double> data_raw(639, 2);
//static int days = 360;
static CIR_params temp;
typedef matrix<double, 0, 1> params_vector;

extern CIR_params getCIRParam(matrix<double, 0, 1> param, matrix<double, 0, 1> tau, int scalingFact);
extern double LogLikCIR(const params_vector& theta);
extern const params_vector num_jacobian(const params_vector& theta);
extern params_vector NewtonRaphson(params_vector& theta, int num_ite = 1000);
extern params_vector EstCIR(matrix<double> data);