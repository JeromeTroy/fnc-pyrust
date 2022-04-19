#[macro_use]

extern crate ndarray;
use ndarray::prelude::*;

/*
    Solve U x = b 
    where U is upper triangular
*/
fn backsub(U : &Array2<f64>, b : &Array1<f64>) -> Array1<f64>{

    let n = U.shape()[0] - 1;
    let mut x : Array1<f64> = Array::zeros((n + 1));

    x[[n]] = b[[n]] / U[[n, n]];

    let mut s = 0.0;
    for index in (0..n).rev() {

        let u_small = U.slice(s![index, index+1..]);
        let x_small = x.slice(s![index+1..]);

        s = u_small.dot(&x_small);
        x[[index]] = (b[[index]] - s) / U[[index, index]];

    }
    return x;
}

fn main() {
    let U = array![[1.0, -1.0, 0.0, -1.9, 2.2], 
                   [0.0, 1.0, -1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, -1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0, -1.0],
                   [0.0, 0.0, 0.0, 0.0, 1.0]];
    
    let x_exact : Array1<f64> = Array::ones(5);
    let b = U.dot(&x_exact);

    let x : Array1<f64> = backsub(&U, &b);
    let error = x - x_exact;

    println!("{}", error);
}