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

#[test]
fn forwardsub_operation() {
    let L = array![[4.0, 0.0, 0.0, 0.0, 0.0],
                   [2.0, 7.0, 0.0, 0.0, 0.0],
                   [1.0, 1.0, 9.0, 0.0, 0.0],
                   [1.0, 1.0, 7.0, 6.0, 0.0],
                   [3.0, 1.0, 9.0, 5.0, 6.0]];
    
    let b : Array1<f64> = Array::ones(5);

    let x = fncrustlib::forwardsub(&L, &b);

    println!("{}", x);

    let error = L.dot(&x) - &b;

    println!("{}", &error);
    
    assert!(error.dot(&error) < 1e-4);
}