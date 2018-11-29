// Estimation of the parameters of the CIR model
//%
//%        dr = alpha(beta - r)dt + sigma sqrt(r) dW
//%
//% with market price of risk q(r) = q1 / sqrt(r) + q2 sqrt(r).The time scale
//% is in years and the units are percentages.
//%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//% Input
//%      data: [R, tau](n x 2), with R : annual bonds yields in percentage,
//%             and tau : maturities in years;
//%      method: 'Hessian' (default), 'exact', 'num';
//%      days: number of days per year(default: 360);
//%      significanceLevel: (95 % default).
//%
//% Output
//%       param : parameters(alpha, beta, sigma, q1, q2) of the model;
//%       error: estimation errors for the given confidence level;
//%       rimp: implied spot rate.
//%
//% Example :
//	%[theta, error, rimp] = EstCIR(DataCIR);
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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
#include <boost/math/distributions/non_central_chi_squared.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <dlib/matrix.h>
#include <dlib/optimization.h>
#include <dlib/global_optimization.h>

using namespace std;
using namespace dlib;


struct CIR_params { matrix<double> A, B; double a, b; };
matrix<double, 0, 1>  R, R0, R1, tau;
matrix<double> data_raw(53,2);
int days=360;
CIR_params temp;
typedef matrix<double, 0, 1> params_vector;

//CIR_params getCIRParam(matrix<double, 0, 1> param, matrix<double, 0, 1> tau, int scalingFact) {
//	//% This function computes the terms A and B for the price of a zero - coupon
//	//% bond under the CIR model.
//	
//	/*cout << param << endl;
//	cout << tau << endl;*/
//	double sqrtScalingFact = sqrt(scalingFact);
//
//	double alpha = param(0);
//	double beta = param(1) / scalingFact;
//	double sigma = param(2) / sqrtScalingFact;
//	double q1 = param(3) / sqrtScalingFact;
//	double q2 = param(4) * sqrtScalingFact;
//		
//	temp.a = alpha + q2 * sigma;
//	temp.b = (alpha * beta - q1 * sigma) / temp.a;
//	double gam = sqrt(temp.a * temp.a + 2 * sigma *sigma);
//	matrix<double, 0, 1> d = 1 - exp(-gam * tau);
//
//	//cout << d << endl;
//
//	temp.B = pointwise_multiply(d, 1 / (0.5 * (gam + temp.a) * d + gam * (1 - d)));
//	
//	temp.A = -(2 * temp.a * temp.b / (sigma * sigma)) * (0.5 * (gam - temp.a) * tau + log(0.5 * (1 + temp.a / gam) * d + (1 - d)));
//	/*cout << "temp.B :\n" << temp.B << endl;
//	cout << "temp.A :\n" << temp.A << endl;*/
//	return temp;
//}
double CIRloglike(const params_vector& theta) {
	double h = 0.00277; // timestep = 1/360 days;
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
		pdf_matrix_nc2chisq(t) = pdf(d_non_central, s(t));
	}
	double lnL = sum(-log(2 * c * pdf_matrix_nc2chisq));
	return lnL;
}

//double LogLikCIR(const params_vector& theta) {
//	/*% This function computes the - Log - likelihoods for the CIR model
//		%
//		%            dr = alpha(beta - r)dt + sigma sqrt(r) dW
//		% with market price of risk q(r) = q1 / sqrt(r) + q2 sqrt(r)
//		%
//		% Input
//		%       theta: parameters of the annualized spot rate;
//	%           R: annual returns in percent of the bonds(n x 1);
//	%         tau: maturities(n x 1) in days;
//	%        days: number of days in a year.
//		%
//		% Output
//		%        LL : -log - likelihoods.*/
//
//
//	int	n = R.nr();
//	int	scalingFact=1;
//	double h=0.00277;// 1/360.
//	matrix<double> LL(n-1,1);
//
//	double alpha = exp(theta(0));
//	double beta = exp(theta(1));
//	double sigma = exp(theta(2));
//	double q1 = theta(3);
//	double q2 = theta(4);
//
//	matrix<double, 0, 1> param = {alpha,beta,sigma,q1,q2};
//	
//
//	getCIRParam(param, tau, scalingFact);
//	if ((temp.a <= 0) || (temp.b <= 0)) {
//		LL = 1.0e+20*LL;
//		return sum(LL);
//	}
//
//	matrix<double, 0, 1> r = pointwise_multiply((pointwise_multiply(tau, R) + scalingFact * temp.A), (1 / temp.B));
//	//cout << r;
//	int flag = 0;
//	for (int i = 0; i < r.nr(); ++i) {
//		if (r(i) <= 0) {flag = 1;}
//	}
//	if (flag==1)
//	{
//		LL = 1.0e+20*LL;
//		return sum(LL);
//	}
//
//	//% parameters for the returns
//	double phi = exp(-alpha*h);
//	double nu = 4*alpha*beta/(sigma*sigma);
//	double omega = beta*(1-phi)/nu;
//	//cout << "nu :\n\n " << nu;
//	r = r/omega;
//	//cout << "r_new :\n" << r;
//	matrix<double, 0, 1> D = rowm(r, range(0,r.nr()-2))*phi;
//	//cout << "D :\n" << D;
//
//	////% log - likelihood function
//	//	%LL = log(omega) + log(abs(B(2:end) . / tau(2:end))) - log(max(PDFNChi2(r(2:end), nu, D), eps));
//	//% Better take the Matlab function ncx2pdf which is more stable numerically.
//				
//	matrix<double, 0, 1> eps(r.nr()-1);
//	eps = 0.000001;
//	matrix<double, 0, 1> pdf_matrix_nc2chisq(r.nr()-1);
//	for (int t = 1; t < r.nr(); ++t) {
//		
//		boost::math::non_central_chi_squared_distribution<> d_non_central(nu, D(t - 1));
//		pdf_matrix_nc2chisq(t-1) = pdf(d_non_central,r(t));
//		
//	}
//	matrix<double, 0, 1> f1 = abs(rowm(temp.B, range(1, r.nr()-1)));
//	
//	matrix<double, 0, 1> f2 = abs(1/rowm(tau, range(1, r.nr()-1)));
//	
//	matrix<double, 0, 1> f3 = log(max_pointwise(pdf_matrix_nc2chisq, eps));
//	
//	//cout << "f3 \n" <<  f3;
//	
//	LL = log(omega) + log(pointwise_multiply(f1,f2 )) - f3;
//	
//	//cout << "LL matrix: \n\n\n\n" << LL;
//	//cout << "SUM(LL): \n" << sum(LL);
//	return sum(LL);
//}

const params_vector num_jacobian(const params_vector& theta) {
//# import numpy as np
//	# Compute the symmetric numerical first order derivatives of a
//		# multivariate function.
//#
//		# Inputs: fct_handle: name of a function returning a N x 1 vector;
//	#         x: point(d x 1) at which the derivatives will be computed;
//	#         prec: percentage of + \ - around x(in fraction).
//#
//		# Output: J(derivatives) (N x d)
	double prec = 0.0001;
	params_vector x2 = theta;
	params_vector x1 = theta;
	params_vector A0(5), B0(5), res(5);
	for (int i = 0; i < theta.nr(); ++i) {

		if (x1(i) != 0)
		{
			x1(i) = x1(i) * (1 - prec);
		}
		else
		{
			x1(i) = -prec;
		}
		
		if (x2(i) != 0) 
		{
			x2(i) = x2(i) * (1 + prec);
		}
		else
		{
			x2(i) = +prec;
		}
	
			A0(i) = CIRloglike(x1);
			B0(i) = CIRloglike(x2);
			if (isinf(A0(i)) || isinf(B0(i))) { res(i) = 0; }
			else { res(i) = (B0(i) - A0(i)) / (x2(i) - x1(i)); }
						
	}
	return res;
}

params_vector NewtonRaphson(params_vector& theta, int num_ite = 1000) {
	std::cout << " Iteration process starting: \n" << CIRloglike(theta);
	double eps = 0.0001;
	double log_target = CIRloglike(theta);
	params_vector h_,h = -log_target / num_jacobian(theta);
	
		for (int i = 0; i < num_ite; ++i)
		{
			std::cout << "\n h: \n" << h << endl;
			if (isnan(h) || std::isinf(h))
			{
				params_vector n = log(abs(h - h_)) / log(2) - 1;
				while (isnan(CIRloglike(theta + n))) {
					theta = theta + n;
				}
				std::cout << "Obj : \n\n" << CIRloglike(theta) << "Theta result : \n\n " << theta;
				return theta;
			}
			if (isnan(CIRloglike(theta + h)) || isinf(CIRloglike(theta + h)))
			{
				params_vector n = log(abs(h - h_)) / log(2) - 1;
				while (isnan(CIRloglike(theta + n))) {
					theta = theta + n;
				}
				std::cout << "Obj : \n\n" << CIRloglike(theta) << "Theta result : \n\n " << theta;
				return theta;
			}
			
			if (abs(sum(h)) < eps) {
				std::cout << "Result: \n" << theta;
				return theta;
			}
			h_ = h;
			theta = theta + h;
			
			std::cout << "\n theta moi: \n\n" << "=======================\n" << theta <<"\n======================="<< endl;

			std::cout << "\n No :" << i + 1 << " \n Objective target :  " << CIRloglike(theta) << endl;
			//if (CIRloglike(theta) < -10e+20) { return theta; }
			
			

			if (CIRloglike(theta) > log_target)
			{
				h = CIRloglike(theta) / num_jacobian(theta);
					
			}
			else
			{
				h = -CIRloglike(theta) / num_jacobian(theta);
				
			}
			
			log_target = CIRloglike(theta);
			std::cout << "h :  \n" << h;
			
		}
		std::cout << "Result: \n" << theta; 
		return theta;
	}
	


//params_vector EstCIR(matrix<double> data) {
//
//	for (int i = 0; i < data_raw.nr(); ++i) {
//		tau(i) = data_raw(i, 0);
//		R(i) = data_raw(i, 1);
//		R1(i) = data_raw(i + 1, 1);
//		if (i < data_raw.nr() - 1) { R0(i) = data_raw(i, 1); }
//	}
//	
//	//%% starting values*/
//	
//	double h = 0.00277;
//	int scalingFact = 1;
//	
//
//	//% Rough estimation in the case of a Feller process
//	/*R1 = R(2:end);
//	R0 = R(1:end - 1);*/
//	double phi0 = mean(pointwise_multiply(R1 - mean(R), R0 - mean(R))) / variance(R);
//
//	double alpha0 = -log(phi0) / h;
//	double beta0 = mean(R);
//	double sigma0 = stddev(R)*sqrt(2 * alpha0 / beta0);
//	double nu0 = 0.25*sigma0 * sigma0 / (alpha0*beta0);
//	params_vector param0 = { alpha0, beta0, sigma0, 0, 0 };
//	params_vector theta0 = log(param0);
//	theta0(3) = 0;
//	theta0(4) = 0;
//	/*%theta0 = [log(2); log(mean(R)); log(1); 0; 0];
//	%param0 = [0.5; 2.55; 0.365; 0.3; 0]; % true parameters*/
//	
//	//% pricing coefficients(computed by changing the scale from percentage % to fraction)
//	getCIRParam(param0, tau, scalingFact);
//	//% get short rate
//	matrix<double, 0, 1> r = pointwise_multiply((pointwise_multiply(tau, R) + scalingFact * temp.A), (1 / temp.B));
//	/*cout << "theta0: \n" << theta0 << endl;
//	cout << "r: \n" << r;*/
//	cout << "Theta0 orginal: \n " << theta0 << endl;
//	
//
//	//find_min_using_approximate_derivatives(bfgs_search_strategy(), objective_delta_stop_strategy(1e-7).be_verbose(),LogLikCIR, theta0, -1);
//	
//	//find_min(bfgs_search_strategy(), objective_delta_stop_strategy(1e-7).be_verbose(), LogLikCIR, num_jacobian, theta0, -1);
//
//	//find_min_bobyqa(LogLikCIR, theta0, 9, uniform_matrix<double>(1, 5, -1e100), uniform_matrix<double>(1, 5, 1e100), 20, 1e-6, 1000);
//	
//	//cout << "Params solution:\n" << theta0 << endl;
//	return  NewtonRaphson(theta0);
//	//return theta0;
//
//}

int main() 
	try {
	
	
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
	R1.set_size(i-1);
	R0.set_size(i - 1);
	tau.set_size(i - 1);
	for (int i = 0; i < data_raw.nr(); ++i) {
		tau(i) = data_raw(i, 0);
		R(i) = data_raw(i, 1);
		R1(i) = data_raw(i + 1, 1);
		if (i < data_raw.nr() - 1) { R0(i) = data_raw(i, 1); }
	}
	params_vector theta0 = { 2,1,1,0,0};
	find_min_using_approximate_derivatives(bfgs_search_strategy(), objective_delta_stop_strategy(1e-7).be_verbose(), CIRloglike, theta0, -1);
	
	//params_vector rls = NewtonRaphson(theta0);
	std::cout << "Params solution:\n" << theta0 << endl;
	//params_vector test = {1.90367,3.07598,6.16055,0.898852,0.607892};
	//std::cout << "\n" <<exp(rls);
return 0;
}
catch (std::exception& e)
{
	std::cout << e.what() << endl;
}