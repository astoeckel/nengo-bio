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
#include <cstddef>
#include <cstdint>
#include <random>

#include <Eigen/Dense>

using namespace Eigen;

/**
 * Structure describing a single Poisson source. Note that Poisson sources are
 * automatically normalized such that the time-average is one. Use gain_min and
 * gain_max to adjust the average value of the input sequence.
 */
struct PoissonSource {
	/**
	 * Seed that should be used for the random number generator.
	 */
	uint32_t seed;

	/**
	 * Average rate of the PoissonSource. This corresponds to 1.0 / lambda.
	 */
	double rate;

	/**
	 * Uniform random gain minimum and maximum.
	 */
	double gain_min, gain_max;

	/**
	 * Time constant of the exponential synapse the PoissonSource is connected
	 * to.
	 */
	double tau;

	/**
	 * Constant offset that is applied to the input. When setting rate to zero,
	 * the offset corresponds to the value of a constant input.
	 */
	double offs;
};

struct GaussianSource {
	/**
	 * Seed that should be used for the random number generator.
	 */
	uint32_t seed;

	/**
	 * Mean.
	 */
	double mu;

	/**
	 * Standard deviation.
	 */
	double sigma;

	/**
	 * Time constant of the exponential synapse the GaussianSource is connected
	 * to.
	 */
	double tau;

	/**
	 * Constant offset that is applied to the input. When setting rate to zero,
	 * the offset corresponds to the value of a constant input.
	 */
	double offs;
};

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
	static constexpr double dt = Parameters::dt / Parameters::ss;
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

	/**
	 * Used internally to support a variety of single-neuron and ensemble-based
	 * experiments.
	 */
	template <typename F>
	static void run_from_functor(F f, uint32_t n_neurons, uint32_t n_samples,
	                             double *state, double *out)
	{
		// Iterate over all steps
		for (size_t i = 0; i < n_samples; i++) {
			// Iterate over all neurons in the population
			for (size_t j = 0; j < n_neurons; j++) {
				// Access to the current state: membrane poential and
				// refractoriness
				Map<VecV> v(&state[j * (n_comp + 1)]);
				double &tref = state[j * (n_comp + 1) + n_comp];

				// Compute the A-matrix and the b-vector for this sample
				VecX x = f(i, j);
				MatA A = calc_A(x);
				VecB b = calc_b(x);

				// Write the initial output value
				out[i * n_neurons + j] = 0.;

				// Advance the simulation for the number of subsamples
				for (size_t s = 0; s < ss; s++) {
					v += (A * v + b) * dt;

					// Clamp the potentials to reasonable values
					v = v.array()
					        .min(Parameters::v_max.array())
					        .max(Parameters::v_min.array());

					// Handle the refractory/spike period for the somatic
					// compartment
					if (tref > 0.0) {
						tref -= dt;
						v[0] = (tref >= tau_ref) ? v_spike : v_reset;
					}

					// Handle spike production
					if (v[0] > v_th && tref <= 0.0) {
						// The neuron spiked, account for the amount of time the
						// neuron already spent in the refractory state since
						// the beginning of the timestep
						tref = tau_ref + tau_spike;
						v[0] = (tau_spike > 0.0) ? v_spike : v_reset;

						// Record the spike
						out[i * n_neurons + j] = dt_inv;
					}
				}
			}
		}
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
	static void run_step_from_memory(uint32_t n_neurons, double *state,
	                                 double *out, const double **xs)
	{
		auto f = [&xs](size_t, size_t neuron) -> VecX {
			VecX x;
			for (size_t k = 0; k < n_inputs; k++) {
				x[k] = xs[k][neuron];
			}
			return x;
		};
		run_from_functor(f, n_neurons, 1, state, out);
	}

	/**
	 * Uses the run_from_functor function to implement a single-neuron
	 * simulation with constant input.
	 */
	static void run_single_with_constant_input(uint32_t n_samples,
	                                           double *state, double *out,
	                                           const double *xs)
	{
		const VecX x(xs);
		auto f = [x](size_t, size_t) -> VecX { return x; };
		run_from_functor(f, 1, n_samples, state, out);
	}

	/**
	 * Uses run_from_functor with a set of Poisson distributed spike sources as
	 * inputs.
	 *
	 * @param sources is a list of PoissonSource descriptors.
	 */
	static void run_single_with_poisson_sources(uint32_t n_samples,
	                                            double *state, double *out,
	                                            const PoissonSource *sources)
	{
		// Initialize the individual random engines for the input channels,
		// pre-compute some filter constants
		std::array<std::mt19937, n_inputs> random_engines;
		std::array<std::exponential_distribution<double>, n_inputs> dist_exp;
		std::array<std::uniform_real_distribution<double>, n_inputs> dist_gain;
		VecX filt, xs, offs, T;
		for (size_t j = 0; j < n_inputs; j++) {
			// Initialize the random engine for this input with the seed
			// specified by the user
			random_engines[j].seed(sources[j].seed);

			// Compute the filter coefficient
			filt[j] = 1.0 - (dt * ss) / sources[j].tau;

			// Setup the poisson distribution and draw the first spike time
			dist_exp[j] =
			    std::exponential_distribution<double>(sources[j].rate);
			T[j] = dist_exp[j](random_engines[j]);

			// Setup the uniform gain distribution and initialize xs to the
			// average value
			const double scale = 1.0 / (sources[j].tau * sources[j].rate);
			dist_gain[j] = std::uniform_real_distribution<double>(
			    sources[j].gain_min * scale, sources[j].gain_max * scale);
			xs[j] = 0.5 * (sources[j].gain_min + sources[j].gain_max);

			// Copy the offset
			offs[j] = sources[j].offs;
		}

		// Implement the Poisson Source
		auto f = [&](size_t i, size_t) {
			// Apply the exponential filter
			xs = xs.array() * filt.array();

			// Generate new input events
			const double curT = i * ss * dt;
			for (size_t j = 0; j < n_inputs; j++) {
				while (T[j] < curT) {
					// Feed a Delta pulse into the input
					xs[j] += dist_gain[j](random_engines[j]);

					// Compute the next spike time
					T[j] += dist_exp[j](random_engines[j]);
				}
			}

			// Return the result
			return xs + offs;
		};

		// Run the actual simulation
		run_from_functor(f, 1, n_samples, state, out);
	}

	/**
	 * Uses run_from_functor with a set of Gaussian noise sources as input.
	 */
	static void run_single_with_gaussian_sources(uint32_t n_samples,
	                                             double *state, double *out,
	                                             const GaussianSource *sources)
	{
		// Initialize the individual random engines for the input channels,
		// pre-compute some filter constants
		std::array<std::mt19937, n_inputs> random_engines;
		std::array<std::normal_distribution<double>, n_inputs> dist_norm;
		VecX filt, xs, offs;
		for (size_t j = 0; j < n_inputs; j++) {
			// Initialize the random engine for this input with the seed
			// specified by the user
			random_engines[j].seed(sources[j].seed);

			// Compute the filter coefficient
			filt[j] = 1.0 - (dt * ss) / sources[j].tau;

			// Setup the poisson distribution and draw the first spike time
			const double scale = (dt * ss) / sources[j].tau;
			dist_norm[j] = std::normal_distribution<double>(
			    scale * sources[j].mu, scale * sources[j].sigma);

			// Initialize xs to the average
			xs[j] = sources[j].mu;

			// Copy the offset
			offs[j] = sources[j].offs;
		}

		// Implement the Poisson Source
		auto f = [&](size_t, size_t) {
			// Apply the exponential filter
			xs = xs.array() * filt.array();

			// Sample the noise source
			for (size_t j = 0; j < n_inputs; j++) {
				xs[j] += dist_norm[j](random_engines[j]);
			}

			// Return the result
			return (xs + offs).cwiseMax(0.0);
		};

		// Run the actual simulation
		run_from_functor(f, 1, n_samples, state, out);
	}
};
}  // namespace
