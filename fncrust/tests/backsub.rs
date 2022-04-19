#[macro_use]

extern crate ndarray;
use ndarray::prelude::*;

use fncrustlib;

#[test]
fn backsub_operation() {
    let U = array![[1.0, -1.0, 0.0, -1.9, 2.2], 
                   [0.0, 1.0, -1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, -1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0, -1.0],
                   [0.0, 0.0, 0.0, 0.0, 1.0]];
    
    let x_exact : Array1<f64> = Array::ones(5);
    let b = U.dot(&x_exact);

    let x : Array1<f64> = fncrustlib::backsub(&U, &b);
    let error = x - x_exact;

    assert!(error.dot(&error) < 1e-4);
}