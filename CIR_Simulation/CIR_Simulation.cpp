// CIR_Simulation.cpp : This file contains the 'main' function. Program execution begins and ends there.
// # Simulation of a Feller process satisfying
//# dr < -alpha(beta - r)dt + sigma sqrt(r) dW
//#
//	#  Input
//	#    r0: initial value
//	#     n : number of values
//	#     h : time step between observations.
//#
//#
//	#   Output
//	#      r: annual rate in percent
//#
//	# r = sim.cir(0.5, 2.55, 0.365, 2.55, 720, 1 / 360)
//

#include "pch.h"
#include <iostream>
#include <math.h>
#include <string>
#include <fstream>
#include <cstdlib>
#include <random>
#include <boost/random/non_central_chi_squared_distribution.hpp>
#include <boost/math/distributions/non_central_chi_squared.hpp>
#include <dlib/matrix.h>
#include <dlib/optimization.h>
#include <dlib/global_optimization.h>
#include <armadillo>

using namespace std;
using namespace dlib;
using namespace arma;
struct CIR_params { matrix<double> A, B; double a, b; };
typedef matrix<double, 0, 1> params_vector;
matrix<double, 0, 1>  R, R0, R1, tau;
matrix<double> data_raw(53, 2);
int days = 360;
double CIRloglike(const params_vector& theta) {
	double h = 1. / days; // timestep = 1/360 days;
	double alpha = exp(theta(0));
	double mu = exp(theta(1));
	double sigma = exp(theta(2));
	double c = 2 * alpha / (sigma*sigma*(1 - exp((-alpha * h))));
	double q = 2 * alpha * mu / (sigma*sigma) - 1;
	params_vector u = c * exp(-alpha * h)*R0;
	params_vector v = c * R1;
	params_vector s = 2 * c * R1;
	params_vector nc = 2 * u;
	matrix<double, 0, 1> pdf_matrix_nc2chisq(s.nr());
	double df = 2 * q + 2;


	for (int t = 0; t < s.nr(); ++t) {
		boost::math::non_central_chi_squared_distribution<> d_non_central(df, nc(t));
		pdf_matrix_nc2chisq(t) = max(pdf(d_non_central, s(t)),1E-15);
	}
	double lnL = sum(-log(2 * c * pdf_matrix_nc2chisq));
	return lnL;
}
matrix<double> sim_cir(double alpha, double  beta, double sigma, double r0, int n, double h, int num_sim) {
	// precomputations
	//	# alpha, beta, sigma, r0, n, h = [0.5, 3, 2, 2.78, 100, 1 / 360]
	matrix<double> r(n,num_sim);

	double	sigmaSquared = sigma * sigma;
	double	nu = 4 * alpha * beta / sigmaSquared;
	double  phi = exp(-alpha * h);
	double  omega = sigmaSquared * (1 - phi) / (4 * alpha);
	std::random_device rd;
	std::default_random_engine eng(rd());
	std::string filename = "CIR_Results.csv";
	ofstream fs;
	fs.open(filename);
	
	for (int i = 0; i < num_sim; ++i) {
		r(0, i) = r0;
		fs << r(0,i) << ",";
		for (int t = 1; t < n; ++t) {
			double x = r(t - 1, i) / omega;
			double D = x * phi;
			boost::random::non_central_chi_squared_distribution<> dist(nu, D);
			double tt = dist(eng);
			r(t, i) = omega * tt;
			fs << omega * tt << ",";
			
		}
		fs << endl;
	}
fs.close();
return r;
}

int main()
	try
		{
	clock_t tStart = clock();
	
	ifstream ip("Raw2.csv");
	if (!ip.is_open())
		std::cout << "ERROR: File Open" << '\n';

	string Tenor;
	string Rate;

	int i = 0;
	while (ip.good()) {
		getline(ip, Tenor, ',');
		getline(ip, Rate, '\n');
		data_raw(i, 0) = std::stod(Tenor);
		data_raw(i, 1) = std::stod(Rate);
		++i;
	}
	ip.close();
	R.set_size(i);
	R1.set_size(i - 1);
	R0.set_size(i - 1);
	tau.set_size(i);
	for (int i = 0; i < data_raw.nr(); ++i) {
		tau(i) = data_raw(i, 0);
		R(i) = data_raw(i, 1);
		R1(i) = data_raw(i + 1, 1);
		if (i < data_raw.nr() - 1) { R0(i) = data_raw(i, 1); }
	}
	double h = 1. / days;
	params_vector dx = pointwise_multiply((R1 - R0),pow(R0, -2));
	params_vector d1 = h * pow(R0, -2);
	params_vector d2 = h * pow(R0, 2);
	matrix<double, 52, 2> regressors;
	arma::mat A(52, 2);
	arma::mat B(52, 1);
	
	//regressors = [d1,d2];
	for (int i = 0; i <regressors.nr(); ++i) {
		//cout << regressors(i,0) << "  " << regressors(i,1) << endl;
		regressors(i, 0) = d1(i);
		regressors(i, 1) = d2(i);
		A(i,0) = d1(i);
		A(i,1) = d2(i);
		B(i) = dx(i);
	}
	for (int i = 0; i < regressors.nr(); ++i) {
		cout << A(i, 0) << "  " << A(i, 1) << endl;
	
	}
	
	cout << solve(A, B) << endl;
	params_vector theta0 = { 1,1,1,0,0 };
	
	find_min_using_approximate_derivatives(bfgs_search_strategy(), objective_delta_stop_strategy(1E-15).be_verbose(), CIRloglike, theta0, -1);
	std::cout << "Params solution:\n" << theta0 << endl;
	//params_vector rls = { exp(theta0(0)),exp(theta0(1)),exp(theta0(2)) };
	// sim_cir(rls(0), rls(1), rls(2), 2.78, 250, 0.0027,100);
	// //std::cout << r;
	 printf("Time completed in : %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	 return 0;
}

catch (std::exception& e)
{
	cout << e.what() << endl;
}

//void run_est()
//
////****************************************************************************80
////
////  Purpose:
////
////    TEST01 demonstrates the use of NELMIN on ROSENBROCK.
////
////  Licensing:
////
////    This code is distributed under the GNU LGPL license. 
////
////  Modified:
////
////    27 February 2008
////
////  Author:
////
////    John Burkardt
////
//{
//	int i;
//	int icount;
//	int ifault;
//	int kcount;
//	int konvge;
//	int n;
//	int numres;
//	double reqmin;
//	double *start;
//	double *step;
//	double *xmin;
//	double ynewlo;
//
//	n = 2;
//
//	start = new double[n];
//	step = new double[n];
//	xmin = new double[n];
//
//	cout << "\n";
//	cout << "TEST01\n";
//	cout << "  Apply NELMIN to ROSENBROCK function.\n";
//
//	start[0] = -1.2;
//	start[1] = 1.0;
//
//	reqmin = 1.0E-08;
//
//	step[0] = 1.0;
//	step[1] = 1.0;
//
//	konvge = 10;
//	kcount = 500;
//
//	cout << "\n";
//	cout << "  Starting point X:\n";
//	cout << "\n";
//	for (i = 0; i < n; i++)
//	{
//		cout << "  " << setw(14) << start[i] << "\n";
//	}
//
//	ynewlo = rosenbrock(start);
//
//	cout << "\n";
//	cout << "  F(X) = " << ynewlo << "\n";
//
//	nelmin(rosenbrock, n, start, xmin, &ynewlo, reqmin, step,
//		konvge, kcount, &icount, &numres, &ifault);
//
//	cout << "\n";
//	cout << "  Return code IFAULT = " << ifault << "\n";
//	cout << "\n";
//	cout << "  Estimate of minimizing value X*:\n";
//	cout << "\n";
//	for (i = 0; i < n; i++)
//	{
//		cout << "  " << setw(14) << xmin[i] << "\n";
//	}
//
//	cout << "\n";
//	cout << "  F(X*) = " << ynewlo << "\n";
//
//	cout << "\n";
//	cout << "  Number of iterations = " << icount << "\n";
//	cout << "  Number of restarts =   " << numres << "\n";
//
//	delete[] start;
//	delete[] step;
//	delete[] xmin;
//
//	return;
//}
////****************************************************************************80