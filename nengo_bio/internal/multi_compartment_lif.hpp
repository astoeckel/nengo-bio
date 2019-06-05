/*
 *  nengo_bio -- Extensions to Nengo for more biological plausibility
 *  Copyright (C) 2019  Andreas Stöckel
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
 
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <random>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace Eigen;

namespace {  // Do not export the following symbols
template <typename Parameters>
class Simulator {
private:
	/*
	 * Somatic constants
	 */
	static constexpr double tau_ref = Parameters::tau_ref;
	static constexpr double tau_spike = Parameters::tau_spike;
	static constexpr double v_th = Parameters::v_th;
	static constexpr double v_reset = Parameters::v_reset;
	static constexpr double v_spike = Parameters::v_spike;

	/*
	 * Constants describing the neuron model
	 */
	static constexpr size_t n_comp = Parameters::n_comp;
	static constexpr size_t n_inputs = Parameters::n_inputs;

	/*
	 * Simulator-specific constants
	 */
	static constexpr size_t ss = Parameters::ss;
	static constexpr double dt = Parameters::dt / ss;
	static constexpr double dt_inv = 1.0 / Parameters::dt;

	/*
	 * Type aliases
	 */
	using MatA = typename Parameters::MatA;
	using VecB = typename Parameters::VecB;
	using VecX = typename Parameters::VecX;
	using VecV = typename Parameters::VecV;

	/**
	 * Returns the A matrix of the linear dynamical system.
	 *
	 * @param x is the vector containing the input data.
	 * @param tar is the target matrix.
	 */
	static MatA calc_A(const VecX &x) { return Parameters::calc_A(x); }

	/**
	 * Returns the b vector of the linear dynamical system.
	 *
	 * @param x is the vector containing the input data.
	 * @param tar is the target matrix.
	 */
	static VecB calc_b(const VecX &x) { return Parameters::calc_b(x); }

	/**
	 * Used internally to quickly compute the matrix exponential given the
	 * current Eigen-decomposition of the matrix.
	 */
	template <typename M1, typename M2>
	static M1 expm(const M1 &Q, const M2 &L, double t)
	{
		const auto Ldiag =
		    DiagonalMatrix<typename M1::Scalar, M2::RowsAtCompileTime>(
		        exp(t * L.array()).matrix());
		return Q * Ldiag * Q.transpose();
	}

public:
	/**
	 * Executes the actual neuron simulation.
	 *
	 * @param xs is an array containing the current neural input. This input is
	 * held constant for all subsamples.
	 * @param state is a pointer at an array of doubles holding the neuron
	 * state. This array must have n_neurons x (m + 1) elements, where m is the
	 * number of compartments in the neuron model.
	 * @param out is the target array that determines whether a neuron
	 * spiked (entry set to true) or did not spike (entry set to false).
	 */
	static void step_math(uint32_t n_neurons, double *xs, double *state, double *out)
	{
		/**
		 * Eigenvalue/eigenvector solver.
		 */
		SelfAdjointEigenSolver<MatA> eigs;

		// Iterate over all neurons in the population
		for (size_t i = 0; i < n_neurons; i++) {
			// Fetch the input data
			const VecX x(&xs[i * n_inputs]);

			// Compute the A-matrix and the b-vector for this sample
			MatA A = calc_A(x);
			VecB b = calc_b(x);

			// Compute the eigenvalues and eigenvectors of the A matrix
			eigs.compute(A);
			const Matrix<double, n_comp, 1> &L = eigs.eigenvalues();
			const Matrix<double, n_comp, n_comp> &Q = eigs.eigenvectors();

			// Compute the equilibrium potential using the
			// eigen-decomposition used above; invert the A matrix by using
			// the reciprocal of the eigenvalues. Note that this is
			// definitively not the best way to solve a linear equation.
			// However, since we have to compute the eigen decomposition
			// anyways, and the matrices used here are relatively small,
			// this should work.
			const Matrix<double, n_comp, 1> Linv = 1.0 / L.array();
			const Matrix<double, n_comp, n_comp> Qinv =
			    Q * Linv.asDiagonal();
			const VecV vEq = -Qinv * Q.transpose() * b;

			// Access to the current state: membrane poential and refractoriness
			Map<VecV> v(&state[i * (n_comp + 1)]);
			double &tref = state[i * (n_comp + 1) + n_comp];

			// Compute the membrane potential for an offset of dt
			const MatA eA = expm(Q, L, dt);

			// Advance the simulation for the number of subsamples
			for (size_t s = 0; s < ss; s++) {
				v = vEq + (eA * (v - vEq));

				// Handle the refractory/spike period for the somatic
				// compartment
				if (tref > 0.0) {
					tref -= dt;
					v[0] = (tref >= tau_ref) ? v_spike : v_reset;
				}

				// Handle spike production
				if (v[0] > v_th && tref <= 0.0) {
					// The neuron spiked, account for the amount of time the
					// neuron already spent in the refractory state since the
					// beginning of the timestep
					tref = tau_ref + tau_spike;
					v[0] = (tau_spike > 0.0) ? v_spike : v_reset;

					// Record the spike
					out[i] = dt_inv;
				}
			}
		}
	}

};
}  // namespace
