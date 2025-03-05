use super::Vector;
use crate::{array_plus_opx, StrError};

/// Performs the addition of two vectors
///
/// ```text
/// w := α⋅u + β⋅v
/// ```
///
/// # Examples
///
/// ```
/// use russell_lab::{vec_add, Vector, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let u = Vector::from(&[10.0, 20.0, 30.0, 40.0]);
///     let v = Vector::from(&[2.0, 1.5, 1.0, 0.5]);
///     let mut w = Vector::new(4);
///     vec_add(&mut w, 0.1, &u, 2.0, &v)?;
///     let correct = "┌   ┐\n\
///                    │ 5 │\n\
///                    │ 5 │\n\
///                    │ 5 │\n\
///                    │ 5 │\n\
///                    └   ┘";
///     assert_eq!(format!("{}", w), correct);
///     Ok(())
/// }
/// ```
pub fn vec_add(w: &mut Vector, alpha: f64, u: &Vector, beta: f64, v: &Vector) -> Result<(), StrError> {
    array_plus_opx(w.as_mut_data(), alpha, u.as_data(), beta, v.as_data())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{vec_add, Vector};
    use crate::{vec_approx_eq, MAX_DIM_FOR_NATIVE_BLAS};

    #[test]
    fn vec_add_fail_on_wrong_dims() {
        let u_2 = Vector::new(2);
        let u_3 = Vector::new(3);
        let v_2 = Vector::new(2);
        let v_3 = Vector::new(3);
        let mut w_2 = Vector::new(2);
        assert_eq!(vec_add(&mut w_2, 1.0, &u_3, 1.0, &v_2), Err("arrays are incompatible"));
        assert_eq!(vec_add(&mut w_2, 1.0, &u_2, 1.0, &v_3), Err("arrays are incompatible"));
    }

    #[test]
    fn vec_add_works() {
        const NOISE: f64 = 1234.567;
        #[rustfmt::skip]
        let u = Vector::from(&[
            1.0, 2.0,
            1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0,
        ]);
        #[rustfmt::skip]
        let v = Vector::from(&[
            0.5, 1.0,
            0.5, 1.0, 1.5, 2.0,
            0.5, 1.0, 1.5, 2.0,
            0.5, 1.0, 1.5, 2.0,
            0.5, 1.0, 1.5, 2.0,
        ]);
        let mut w = Vector::from(&vec![NOISE; u.dim()]);
        vec_add(&mut w, 1.0, &u, -4.0, &v).unwrap();
        #[rustfmt::skip]
        let correct = &[
            -1.0, -2.0,
            -1.0, -2.0, -3.0, -4.0,
            -1.0, -2.0, -3.0, -4.0,
            -1.0, -2.0, -3.0, -4.0,
            -1.0, -2.0, -3.0, -4.0,
        ];
        vec_approx_eq(&w, correct, 1e-15);
    }

    #[test]
    fn vec_add_sizes_works() {
        const NOISE: f64 = 1234.567;
        for size in 0..(MAX_DIM_FOR_NATIVE_BLAS + 3) {
            let mut u = Vector::new(size);
            let mut v = Vector::new(size);
            let mut w = Vector::from(&vec![NOISE; u.dim()]);
            let mut correct = vec![0.0; size];
            for i in 0..size {
                u[i] = i as f64;
                v[i] = i as f64;
                correct[i] = i as f64;
            }
            vec_add(&mut w, 0.5, &u, 0.5, &v).unwrap();
            vec_approx_eq(&w, &correct, 1e-15);
        }
    }
}
