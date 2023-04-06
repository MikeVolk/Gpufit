#ifndef HYPERFINE_CUH_INCLUDED
#define HYPERFINE_CUH_INCLUDED

/* Description of the calculate_superposed_doublet function
 * =========================================================
 *
 * This function calculates the values of the superposition of two doublets,
 * each consisting of two one-dimensional Lorentzian model functions,
 * and their partial derivatives with respect to the model parameters.
 *
 * Refer to the documentation of the calculate_esr3rt function for details
 * on how the X values are handled.
 *
 * Parameters:
 *
 * parameters: An input vector of model parameters.
 *             p[0]: center coordinate of the doublets
 *             p[1]: symmetric shift between the left and right doublets
 *             p[2]: width of the Lorentzians in both doublets
 *             p[3]: amplitude of the Lorentzians in the left doublet
 *             p[4]: amplitude of the Lorentzians in the right doublet
 *             p[5]: offset
 *
 * n_fits: The number of fits. (not used)
 *
 * n_points: The number of data points per fit.
 *
 * value: An output vector of model function values.
 *
 * derivative: An output vector of model function partial derivatives.
 *
 * point_index: The data point index.
 *
 * fit_index: The fit index. (not used)
 *
 * chunk_index: The chunk index. (not used)
 *
 * user_info: An input vector containing user information.
 *
 * user_info_size: The size of user_info in bytes.
 *
 * Calling the calculate_superposed_doublet function
 * =================================================
 *
 * This __device__ function can be only called from a __global__ function or another
 * __device__ function.
 *
 */
__device__ void calculate_superposed_doublet(
    float const *parameters,
    int const n_fits,
    int const n_points,
    float *value,
    float *derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char *user_info,
    std::size_t const user_info_size)
{
    // INDICES

    float *user_info_float = (float *)user_info;
    float x = 0.0f;
    if (!user_info_float)
    {
        x = point_index;
    }
    else if (user_info_size / sizeof(float) == n_points)
    {
        x = user_info_float[point_index];
    }
    else if (user_info_size / sizeof(float) > n_points)
    {
        int const chunk_begin = chunk_index * n_fits * n_points;
        int const fit_begin = fit_index * n_points;
        x = user_info_float[chunk_begin + fit_begin + point_index];
    }

    // PARAMETERS
    float const *p = parameters;

    // LORENTZIAN FUNCTION CALCULATIONS
    float const ahyp = 0.0015;

    float center_left = p[0] - p[1];
    float center_right = p[0] + p[1];

    float center_left_1 = x - center_left - ahyp;
    float center_left_2 = x - center_left + ahyp;

    float center_right_1 = x - center_right - ahyp;
    float center_right_2 = x - center_right + ahyp;

    float width_sq = p[2] * p[2];

    float peak_left_left = p[2] / (center_left_1 * center_left_1 + width_sq);
    float peak_left_right = p[2] / (center_left_2 * center_left_2 + width_sq);
    float peak_right_left = p[2] / (center_right_1 * center_right_1 + width_sq);
    float peak_right_right = p[2] / (center_right_2 * center_right_2 + width_sq);

    float peak_left_left_sq = peak_left_left * peak_left_left;
    float peak_left_right_sq = peak_left_right * peak_left_right;
    float peak_right_left_sq = peak_right_left * peak_right_left;
    float peak_right_right_sq = peak_right_right * peak_right_right;

    // VALUE CALCULATION

    value[point_index] = 1 + p[5] - p[3] * (peak_left_left + peak_left_right) - p[4] * (peak_right_left + peak_right_right);

    // DERIVATIVE CALCULATION
    float *current_derivative = derivative + point_index;

    current_derivative[0 * n_points] = 2 * p[1] * (p[3] * (center_left_1 * peak_left_left_sq + center_left_2 * peak_left_right_sq) + p[4] * (center_right_1 * peak_right_left_sq + center_right_2 * peak_right_right_sq));
    current_derivative[1 * n_points] = -2 * (p[3] * (center_left_1 * peak_left_left_sq - center_left_2 * peak_left_right_sq) + p[4] * (center_right_1 * peak_right_left_sq - center_right_2 * peak_right_right_sq));
    current_derivative[2 * n_points] = -2 * p[2] * (p[3] * (peak_left_left - peak_left_left_sq + peak_left_right - peak_left_right_sq) + p[4] * (peak_right_left - peak_right_left_sq + peak_right_right - peak_right_right_sq));
    current_derivative[3 * n_points] = -(peak_left_left + peak_left_right);
    current_derivative[4 * n_points] = -(peak_right_left + peak_right_right);
    current_derivative[5 * n_points] = 1.f;
}
#endif