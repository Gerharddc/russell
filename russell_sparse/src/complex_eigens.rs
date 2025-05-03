use crate::{
    solver_arpack::{
        ArpackComplex64, ArpackConfig, ArpackDriver, ArpackMode, ArpackResults, EigenValueProblem, EigenvaluePosition,
    },
    ComplexCooMatrix, ComplexCscMatrix, ComplexLinSolver, Genie, StrError,
};
use russell_lab::{Complex64, NumVector};
use static_assertions::{assert_eq_align, assert_eq_size};

// Just making sure that Complex64 and ArpackComplex64 are indeed equivalent
//
// FIXME: think is this is really useful
assert_eq_size!(Complex64, ArpackComplex64);
assert_eq_align!(Complex64, ArpackComplex64);

impl From<Complex64> for ArpackComplex64 {
    fn from(value: Complex64) -> Self {
        ArpackComplex64 {
            re: value.re,
            im: value.im,
        }
    }
}

impl From<ArpackComplex64> for Complex64 {
    fn from(value: ArpackComplex64) -> Self {
        Complex64 {
            re: value.re,
            im: value.im,
        }
    }
}

fn vec_from_arr(value: &[ArpackComplex64]) -> NumVector<Complex64> {
    // FIXME: if we can verify that ArpackComplex64 is exactly the same as Complex64 in memory
    // then we could just use mem::transmute to perform this conversion much more efficiently.

    NumVector::initialized(value.len(), |i| value[i].into())
}

fn copy_vec_to_arr(arr: &mut [ArpackComplex64], vec: NumVector<Complex64>) {
    // FIXME: if we can verify that ArpackComplex64 is exactly the same as Complex64 in memory
    // then we could just use mem::transmute to perform this conversion much more efficiently.

    for (i, val) in vec.into_iter().enumerate() {
        arr[i] = val.into();
    }
}

/// Type of eigenvaules to compute.
#[derive(Debug, Clone, Copy)]
pub enum Sigma {
    /// Eigenvalues of largest magnitude.
    LargestMagnitude,
    /// Eigenvalues of smallest magnitude.
    SmallestMagnitude,
    /// Eigenvalues of largest real part.
    LargestReal,
    /// Eigenvalues of smallest real part.
    SmallestReal,
    /// Eigenvalues of largest imaginary part.
    LargestImaginary,
    /// Eigenvalues of smallest imaginary part.
    SmallestImaginary,
    /// Eigenvalues closest to a scalar value.
    Scalar(Complex64),
}

impl Sigma {
    fn which(&self) -> EigenvaluePosition {
        match self {
            Sigma::LargestMagnitude => EigenvaluePosition::LargestMagnitude,
            Sigma::SmallestMagnitude => EigenvaluePosition::SmallestMagnitude,
            Sigma::LargestReal => EigenvaluePosition::LargestReal,
            Sigma::SmallestReal => EigenvaluePosition::SmallestReal,
            Sigma::LargestImaginary => EigenvaluePosition::LargestImaginary,
            Sigma::SmallestImaginary => EigenvaluePosition::SmallestImaginary,
            Sigma::Scalar(_) => EigenvaluePosition::LargestMagnitude,
        }
    }

    fn shift(&self) -> Complex64 {
        match self {
            Sigma::LargestMagnitude => Complex64::ZERO,
            Sigma::SmallestMagnitude => Complex64::ZERO,
            Sigma::LargestReal => Complex64::ZERO,
            Sigma::SmallestReal => Complex64::ZERO,
            Sigma::LargestImaginary => Complex64::ZERO,
            Sigma::SmallestImaginary => Complex64::ZERO,
            Sigma::Scalar(value) => *value,
        }
    }
}

pub fn eigens(
    a_mat: &ComplexCooMatrix,
    m_mat: Option<&ComplexCooMatrix>,
    num_eigenvalues: usize,
    sigma: Sigma,
    genie: Genie,
) -> Result<ArpackResults, StrError> {
    let config = ArpackConfig {
        bmat: match &m_mat {
            Some(_) => EigenValueProblem::Generalized,
            None => EigenValueProblem::Standard,
        },
        n: a_mat.nrow as i32,
        which: sigma.which(),
        nev: num_eigenvalues as i32,
        tol: 0.0,                // TODO: verify this is a reasonable default
        mxiter: 1000,            // TODO: verify this is a reasonable default
        ncv: 20,                 // TODO: verify this is a reasonable default
        mode: ArpackMode::Mode3, // TODO: only use shift-and-invert mode if really needed
        shift: sigma.shift().into(),
    };

    let state = DriverState::new(a_mat, m_mat, sigma.shift(), genie)?;
    let mut driver = ArpackDriver::new(config, mx_product, solve_from_x, solve_from_mx, state);
    driver.solve()
}

struct DriverState<'a> {
    solver: ComplexLinSolver<'a>,
    m_mat: Option<ComplexCscMatrix>,
    sigma: Complex64,
}

impl<'a> DriverState<'a> {
    fn new(
        a_mat: &'a ComplexCooMatrix,
        m_mat: Option<&'a ComplexCooMatrix>,
        sigma: Complex64,
        genie: Genie,
    ) -> Result<Self, StrError> {
        // 1. Form the shifted matrix (A-sigma*M)
        // 2. Factor this matrix (could cache this if sigma doesn't change)

        let mut solver = ComplexLinSolver::new(genie)?;
        solver.actual.factorize(&a_mat, None)?;
        //let mut f = Vector::new(n_x);
        //solver.actual.solve(&mut f, &b_vec, false)?;

        let m_mat = match m_mat {
            Some(coo) => Some(ComplexCscMatrix::from_coo(coo)?),
            None => None,
        };

        Ok(Self { solver, m_mat, sigma })
    }
}

/// Perform  y <--- M*x (where x is provided as input).
///
/// # Parameters:
/// * `x`: Input vector
/// * `y`: Output vector
fn mx_product(state: &mut DriverState<'_>, x: &[ArpackComplex64], y: &mut [ArpackComplex64]) {
    match &state.m_mat {
        Some(m_mat) => {
            let x = vec_from_arr(x);
            let mut v = NumVector::new(x.dim());
            m_mat.mat_vec_mul(&mut v, Complex64::ONE, &x).unwrap();
            copy_vec_to_arr(y, v);
        }
        None => {
            // M is an identity matrix which has no effect
            y.clone_from_slice(x);
        }
    }
}

/// Perform y <--- OP*x = inv[A-sigma*M]*M*x (where x is provided as input).
///
/// # Parameters:
/// * `x`: Input vector
/// * `y`: Output vector
fn solve_from_x(state: &mut DriverState<'_>, x: &[ArpackComplex64], y: &mut [ArpackComplex64]) {
    todo!("Implement linear system solver");
}

/// Perform y <--- OP*x = inv[A-sigma*M]*M*x (where x is provided as input).
///
/// # Parameters:
/// * `mx`: Input vector
/// * `y`: Output vector
fn solve_from_mx(state: &mut DriverState<'_>, mx: &[ArpackComplex64], y: &mut [ArpackComplex64]) {
    todo!("Implement linear system solver");
}
