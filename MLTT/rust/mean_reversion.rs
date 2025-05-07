use ndarray::{Array1, Array2, ArrayD, ArrayView2, Axis, s};
use pyo3::prelude::*;
use std::cmp::max;
use std::collections::HashSet;

// Import from allocation_o2
use allocation_o2::allocation::traits::AllocationStrategy;
use allocation_o2::register_strategy;
use allocation_o2::allocation::py_bindings::{numpy_to_ndarray, ndarray_to_numpy};

/// Change function calculates the absolute change in prices over a specified lag period
#[inline(always)]
fn change(prices: &ArrayView2<f64>, lag: usize) -> Array2<f64> {
    let n_rows = prices.shape()[0];
    let n_cols = prices.shape()[1];
    
    // Early return for invalid input
    if n_rows <= lag {
        return Array2::<f64>::zeros((0, n_cols));
    }
    
    let result_rows = n_rows - lag;
    let mut result = Array2::<f64>::zeros((result_rows, n_cols));
    
    // Fast change calculation with direct indexing
    for i in 0..result_rows {
        let row_now = prices.slice(s![i, ..]);
        let row_future = prices.slice(s![i + lag, ..]);
        
        for j in 0..n_cols {
            result[[i, j]] = row_future[j] - row_now[j];
        }
    }
    
    result
}

/// Calculate standard deviation of a slice
#[inline(always)]
fn std_dev(slice: &ArrayView2<f64>, axis: usize, ddof: usize) -> Array1<f64> {
    let n = slice.shape()[axis];
    
    if n <= ddof {
        let mut result = Array1::<f64>::zeros(slice.shape()[1 - axis]);
        result.fill(f64::NAN);
        return result;
    }
    
    if axis == 0 {
        // Column-wise standard deviation (faster path)
        let n_cols = slice.shape()[1];
        let mut result = Array1::<f64>::zeros(n_cols);
        let denom = (n - ddof) as f64;
        
        // Calculate means first
        let mut means = vec![0.0; n_cols];
        
        // Calculate means
        for j in 0..n_cols {
            let mut sum = 0.0;
            for i in 0..n {
                sum += slice[[i, j]];
            }
            means[j] = sum / (n as f64);
        }
        
        // Calculate variance
        for j in 0..n_cols {
            let mut sum_sq = 0.0;
            for i in 0..n {
                let diff = slice[[i, j]] - means[j];
                sum_sq += diff * diff;
            }
            result[j] = (sum_sq / denom).sqrt();
        }
        
        result
    } else {
        // Row-wise standard deviation
        let n_rows = slice.shape()[0];
        let n_cols = slice.shape()[1];
        let mut result = Array1::<f64>::zeros(n_rows);
        let denom = (n - ddof) as f64;
        
        for i in 0..n_rows {
            let row = slice.slice(s![i, ..]);
            
            // Calculate mean
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            
            for j in 0..n_cols {
                sum += row[j];
            }
            
            let mean = sum / (n_cols as f64);
            
            for j in 0..n_cols {
                let diff = row[j] - mean;
                sum_sq += diff * diff;
            }
            
            result[i] = (sum_sq / denom).sqrt();
        }
        
        result
    }
}

/// Enhanced Cross-Sectional Mean Reversion Model with Volatility Filter.
/// Based on the article by Teddy Koker.
#[pyclass]
pub struct EnhancedCrossSectionalMRModel {
    #[pyo3(get, set)]
    num_positions: usize,
    
    #[pyo3(get, set)]
    lag: usize,
    
    #[pyo3(get, set)]
    volatility_period: usize,
}

#[pymethods]
impl EnhancedCrossSectionalMRModel {
    #[new]
    fn new(num_positions: usize, lag: usize, volatility_period: usize) -> Self {
        EnhancedCrossSectionalMRModel {
            num_positions,
            lag,
            volatility_period,
        }
    }
    
    #[getter]
    fn min_observations(&self) -> usize {
        max(self.lag, self.volatility_period)
    }
    
    fn predict(&self, py: Python, input: &PyAny) -> PyResult<PyObject> {
        // Convert input from Python to Rust
        let input_array = numpy_to_ndarray(py, input)?;
        
        // Perform the prediction
        let output_array = self.predict_impl(&input_array);
        
        // Convert the result back to Python
        ndarray_to_numpy(py, output_array)
    }
}

impl AllocationStrategy for EnhancedCrossSectionalMRModel {
    fn min_observations(&self) -> usize {
        max(self.lag, self.volatility_period)
    }
    
    fn predict(&self, input: &ArrayD<f64>) -> ArrayD<f64> {
        self.predict_impl(input)
    }
}

impl EnhancedCrossSectionalMRModel {
    #[inline(always)]
    fn predict_impl(&self, input: &ArrayD<f64>) -> ArrayD<f64> {
        // Convert input to 2D array view for processing
        let shape = input.shape();
        let x = input.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        
        // Quick check for insufficient data
        if shape[0] <= self.lag {
            return ArrayD::zeros(vec![shape[0], shape[1]]);
        }
        
        // Calculate returns using optimized change function
        let returns = change(&x, self.lag);
        let returns_rows = returns.shape()[0];
        
        // Pre-allocate the result
        let mut final_weights = Array2::<f64>::zeros((shape[0], shape[1]));
        
        // Fast path for no returns
        if returns_rows == 0 {
            return final_weights.into_dyn();
        }
        
        // Create padded returns - directly map from returns to padded space
        let mut padded_returns = Array2::<f64>::zeros((shape[0], shape[1]));
        
        // Copy returns into padded_returns
        for i in 0..returns_rows {
            let src_row = returns.row(i);
            for j in 0..shape[1] {
                padded_returns[[i + self.lag, j]] = src_row[j];
            }
        }
        
        // Calculate mean returns for each time step (row means)
        let row_means = padded_returns.mean_axis(Axis(1)).unwrap();
        
        // Calculate weights in one pass (mean - value = negative deviation for mean reversion)
        let mut weights = Array2::<f64>::zeros((shape[0], shape[1]));
        
        // Calculate deviations and weights
        for i in 0..padded_returns.shape()[0] {
            let mean = row_means[i];
            for j in 0..padded_returns.shape()[1] {
                // Negative deviation (mean reversion strategy)
                weights[[i, j]] = mean - padded_returns[[i, j]];
            }
        }
        
        // Pre-calculate volatilities for all time steps
        let mut volatilities = Vec::with_capacity(shape[0]);
        for t in 0..shape[0] {
            if t < self.volatility_period {
                if t > 0 {
                    volatilities.push(std_dev(&x.slice(s![0..t+1, ..]), 0, 1));
                } else {
                    volatilities.push(Array1::<f64>::ones(shape[1]));
                }
            } else {
                volatilities.push(std_dev(&x.slice(s![t-self.volatility_period+1..t+1, ..]), 0, 1));
            }
        }
            
        // Create a HashSet for faster intersection lookups
        let mut top_vol_set = HashSet::with_capacity(self.num_positions);
        
        // Process each time step
        for t in 0..weights.shape()[0] {
            let curr_weights_row = weights.row(t);
            let vol = &volatilities[t];
            
            // Pre-allocate vectors with capacity
            let mut weight_index_pairs = Vec::with_capacity(shape[1]);
            let mut vol_index_pairs = Vec::with_capacity(shape[1]);
            
            // Build pairs for sorting
            for j in 0..shape[1] {
                weight_index_pairs.push((j, curr_weights_row[j].abs()));
                vol_index_pairs.push((j, vol[j]));
            }
            
            // Use faster unstable sort
            weight_index_pairs.sort_unstable_by(|a, b| 
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            vol_index_pairs.sort_unstable_by(|a, b| 
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            
            // Get top positions by lowest volatility (into HashSet for O(1) lookups)
            top_vol_set.clear();
            for i in 0..std::cmp::min(self.num_positions, vol_index_pairs.len()) {
                top_vol_set.insert(vol_index_pairs[i].0);
            }
            
            // Accumulate selected weights directly
            let mut selected = Vec::with_capacity(self.num_positions);
            let mut selected_weights = Vec::with_capacity(self.num_positions);
            let mut norm_factor = 0.0;
            
            // Find intersection directly with HashSet for O(1) lookups
            for i in 0..std::cmp::min(self.num_positions, weight_index_pairs.len()) {
                let idx = weight_index_pairs[i].0;
                if top_vol_set.contains(&idx) {
                    let weight = curr_weights_row[idx];
                    selected.push(idx);
                    selected_weights.push(weight);
                    norm_factor += weight.abs();
                }
            }
            
            // Apply normalized weights
            if norm_factor > 0.0 {
                for (i, &idx) in selected.iter().enumerate() {
                    final_weights[[t, idx]] = selected_weights[i] / norm_factor;
                }
            }
        }
        
        // Convert to dynamic array and return
        final_weights.into_dyn()
    }
}

#[pymodule]
fn mean_reversion(_py: Python, m: &PyModule) -> PyResult<()> {
    register_strategy!(m, EnhancedCrossSectionalMRModel);
    Ok(())
}
