use super::Matrix;
use crate::{array_plus_opx, StrError};

/// Performs the addition of two matrices
///
/// ```text
/// c := α⋅a + β⋅b
/// ```
///
/// # Examples
///
/// ```
/// use russell_lab::{mat_add, Matrix, StrError};
///
/// fn main() -> Result<(), StrError> {
///     let a = Matrix::from(&[
///         [ 10.0,  20.0,  30.0,  40.0],
///         [-10.0, -20.0, -30.0, -40.0],
///     ]);
///     let b = Matrix::from(&[
///         [ 2.0,  1.5,  1.0,  0.5],
///         [-2.0, -1.5, -1.0, -0.5],
///     ]);
///     let mut c = Matrix::new(2, 4);
///     mat_add(&mut c, 0.1, &a, 2.0, &b)?;
///     let correct = "┌             ┐\n\
///                    │  5  5  5  5 │\n\
///                    │ -5 -5 -5 -5 │\n\
///                    └             ┘";
///     assert_eq!(format!("{}", c), correct);
///     Ok(())
/// }
/// ```
pub fn mat_add(c: &mut Matrix, alpha: f64, a: &Matrix, beta: f64, b: &Matrix) -> Result<(), StrError> {
    let (m, n) = c.dims();
    if a.nrow() != m || a.ncol() != n || b.nrow() != m || b.ncol() != n {
        return Err("matrices are incompatible");
    }
    array_plus_opx(c.as_mut_data(), alpha, a.as_data(), beta, b.as_data())
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{mat_add, Matrix};
    use crate::mat_approx_eq;

    #[test]
    fn mat_add_fail_on_wrong_dims() {
        let a_2x2 = Matrix::new(2, 2);
        let a_2x3 = Matrix::new(2, 3);
        let a_3x2 = Matrix::new(3, 2);
        let b_2x2 = Matrix::new(2, 2);
        let b_2x3 = Matrix::new(2, 3);
        let b_3x2 = Matrix::new(3, 2);
        let mut c_2x2 = Matrix::new(2, 2);
        assert_eq!(
            mat_add(&mut c_2x2, 1.0, &a_2x3, 1.0, &b_2x2),
            Err("matrices are incompatible")
        );
        assert_eq!(
            mat_add(&mut c_2x2, 1.0, &a_3x2, 1.0, &b_2x2),
            Err("matrices are incompatible")
        );
        assert_eq!(
            mat_add(&mut c_2x2, 1.0, &a_2x2, 1.0, &b_2x3),
            Err("matrices are incompatible")
        );
        assert_eq!(
            mat_add(&mut c_2x2, 1.0, &a_2x2, 1.0, &b_3x2),
            Err("matrices are incompatible")
        );
    }

    #[test]
    fn mat_add_works() {
        const NOISE: f64 = 1234.567;
        #[rustfmt::skip]
        let a = Matrix::from(&[
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ]);
        #[rustfmt::skip]
        let b = Matrix::from(&[
            [0.5, 1.0, 1.5, 2.0],
            [0.5, 1.0, 1.5, 2.0],
            [0.5, 1.0, 1.5, 2.0],
        ]);
        let mut c = Matrix::from(&[
            [NOISE, NOISE, NOISE, NOISE],
            [NOISE, NOISE, NOISE, NOISE],
            [NOISE, NOISE, NOISE, NOISE],
        ]);
        mat_add(&mut c, 1.0, &a, -4.0, &b).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [-1.0, -2.0, -3.0, -4.0],
            [-1.0, -2.0, -3.0, -4.0],
            [-1.0, -2.0, -3.0, -4.0],
        ];
        mat_approx_eq(&c, correct, 1e-15);
    }

    #[test]
    fn add_matrix_oblas_works() {
        const NOISE: f64 = 1234.567;
        let a = Matrix::from(&[
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ]);
        let b = Matrix::from(&[
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [0.5, 1.0, 1.5, 2.0, 2.5],
        ]);
        let mut c = Matrix::from(&[
            [NOISE, NOISE, NOISE, NOISE, NOISE],
            [NOISE, NOISE, NOISE, NOISE, NOISE],
            [NOISE, NOISE, NOISE, NOISE, NOISE],
            [NOISE, NOISE, NOISE, NOISE, NOISE],
            [NOISE, NOISE, NOISE, NOISE, NOISE],
        ]);
        mat_add(&mut c, 1.0, &a, -4.0, &b).unwrap();
        #[rustfmt::skip]
        let correct = &[
            [-1.0, -2.0, -3.0, -4.0, -5.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0],
        ];
        mat_approx_eq(&c, correct, 1e-15);
    }

    #[test]
    fn mat_add_skip() {
        let a = Matrix::new(0, 0);
        let b = Matrix::new(0, 0);
        let mut c = Matrix::new(0, 0);
        mat_add(&mut c, 1.0, &a, 1.0, &b).unwrap();
        assert_eq!(a.as_data().len(), 0);
    }
}
