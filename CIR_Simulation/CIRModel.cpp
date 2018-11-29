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
#include <armadillo>
# include "asa047.hpp"

using namespace std;
using namespace arma;


arma::mat data_raw(1000, 2);
arma::mat R, tau, R0, R1;
int days = 360;

double CIRloglike(double theta[3]) {
	double h = 1. / days; // timestep = 1/360 days;
	double alpha = theta[0];
	double mu = theta[1];
	double sigma = theta[2];
	double c = 2 * alpha / (sigma * sigma * (1 - exp((-alpha * h))));
	double q = 2 * alpha * mu / (sigma * sigma) - 1;
	arma::vec u = c * exp(-alpha * h) * R0;
	arma::vec v = c * R1;
	arma::vec s = 2 * c * R1;
	arma::vec nc = 2 * u;
	arma::vec pdf_matrix_nc2chisq(s.n_rows);
	double df = 2 * q + 2;


	for (int t = 0; t < s.n_rows; ++t) {
		boost::math::non_central_chi_squared_distribution<> d_non_central(df, nc(t));
		pdf_matrix_nc2chisq(t) = pdf(d_non_central, s(t));
	}
	double lnL = accu(-log(2 * c * pdf_matrix_nc2chisq));
	return lnL;
}

arma::mat sim_cir(double alpha, double beta, double sigma, double r0, int n, double h, int num_sim) {
	// precomputations
	//	# alpha, beta, sigma, r0, n, h = [0.5, 3, 2, 2.78, 100, 1 / 360]
	arma::mat r(n, num_sim);

	double sigmaSquared = sigma * sigma;
	double nu = 4 * alpha * beta / sigmaSquared;
	double phi = exp(-alpha * h);
	double omega = sigmaSquared * (1 - phi) / (4 * alpha);
	std::random_device rd;
	std::default_random_engine eng(rd());
	std::string filename = "CIR_Results.csv";
	ofstream fs;
	fs.open(filename);

	for (int i = 0; i < num_sim; ++i) {
		r(0, i) = r0;
		fs << r(0, i) << ",";
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
void est_cir()

//****************************************************************************80
//
//  Purpose:
//
//    TEST01 demonstrates the use of NELMIN on ROSENBROCK.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    27 February 2008
//
//  Author:
//
//    John Burkardt
//
{
	int i;
	int icount;
	int ifault;
	int kcount;
	int konvge;
	int n;
	int numres;
	double reqmin;
	double *start;
	double *step;
	double *xmin;
	double ynewlo;

	n = 3;

	start = new double[n];
	step = new double[n];
	xmin = new double[n];

	cout << "\n";
	cout << "Estimation params of CIR model with Nelder Mead Algorithm\n";
	cout << "  Apply NELMIN to CIR_Log_like function.\n";

	start[0] = 3.0;
	start[1] = 2.0;
	start[2] = 1.0;

	reqmin = 1.0E-08;

	step[0] = 0.1;
	step[1] = 0.1;
	step[2] = 0.1;

	konvge = 10;
	kcount = 500;

	cout << "\n";
	cout << "  Starting point X:\n";
	cout << "\n";
	for (i = 0; i < n; i++)
	{
		cout << "  " << setw(14) << start[i] << "\n";
	}

	ynewlo = CIRloglike(start);

	cout << "\n";
	cout << "  F(X) = " << ynewlo << "\n";

	nelmin(CIRloglike, n, start, xmin, &ynewlo, reqmin, step,
		konvge, kcount, &icount, &numres, &ifault);

	cout << "\n";
	cout << "  Return code IFAULT = " << ifault << "\n";
	cout << "\n";
	cout << "  Estimate of minimizing value X*:\n";
	cout << "\n";
	for (i = 0; i < n; i++)
	{
		cout << "  " << setw(14) << xmin[i] << "\n";
	}

	cout << "\n";
	cout << "  F(X*) = " << ynewlo << "\n";

	cout << "\n";
	cout << "  Number of iterations = " << icount << "\n";
	cout << "  Number of restarts =   " << numres << "\n";
	sim_cir(start[0], start[1], start[2], 4.13, 90, 0.00278, 100);
	delete[] start;
	delete[] step;
	delete[] xmin;

	return;
}
//****************************************************************************80


int main() {
	try {
		clock_t tStart = clock();

		ifstream ip("data.csv");
		if (!ip.is_open())
			std::cout << "ERROR: File Open" << '\n';

		string Tenor;
		string Rate;

		int i = 0;
		double h = 1. / days;
		while (ip.good()) {
			getline(ip, Tenor, ',');
			getline(ip, Rate, '\n');
			data_raw(i, 0) = std::stod(Tenor);
			data_raw(i, 1) = std::stod(Rate);
			++i;
		}
		ip.close();
		   // data_raw.set_size(i, 2);
			//R.set_size(i);
			R1.set_size(i - 1);
			R0.set_size(i - 1);
			//tau.set_size(i);
		for (int t = 0; t < i; ++t) {
			//tau(t) = data_raw(t, 0);
			//R(t) = data_raw(t, 1);
			if (t < i - 1)
			{
				R1(t) = data_raw(t + 1, 1);
				R0(t) = data_raw(t, 1);
			}
		}
		data_raw.reset();

		arma::mat dx = (R0 - R1) / pow(R0, 0.5);
		arma::mat regressors(R0.n_rows, 2);
		regressors.col(0) = h / pow(R0, 0.5);
		regressors.col(1) = h * pow(R0, 0.5);

		arma::mat drift;
		//    drift = solve(regressors,dx,solve_opts::allow_ugly + solve_opts::no_approx);
		solve(drift, regressors, dx, solve_opts::allow_ugly + solve_opts::no_approx);
		arma::mat U, V;
		arma::vec s;

		arma::svd(U, s, V, regressors, "std");
		// cout << "U matrix:"  << U << endl;
		cout << "s vector:" << s << endl;
		cout << "V matrix:" << V << endl;
		arma::mat res = regressors * drift - dx;
		//cout << res << endl;

		double alpha = drift(1);
		double mu = -drift(0) / drift(1);
		double sigma = sqrt(var(res.col(0)) / h);
		cout << alpha << "  " << mu << "  " << sigma;
		double params[] = { alpha,mu,sigma };
		cout << "\n" << CIRloglike(params);

	}
	catch (std::exception& e) {
		cout << e.what() << endl;
	}
	est_cir(); // add sim process at the of estimation process
	return 0;
}

