use russell_lab::Complex64;

use crate::{
    solver_arpack::{
        ArpackComplex64, ArpackConfig, ArpackDriver, ArpackMode, ArpackResults, EigenValueProblem, EigenvaluePosition,
    },
    ComplexCooMatrix, ComplexLinSolver, Genie, StrError,
};

pub struct ComplexDriver<'a> {
    solver: ComplexLinSolver<'a>,
}

impl<'a> ComplexDriver<'a> {
    fn new(a_mat: &ComplexCooMatrix, genie: Genie) -> Result<Self, StrError> {
        let mut solver = ComplexLinSolver::new(genie)?;
        solver.actual.factorize(&a_mat, None)?;
        //let mut f = Vector::new(n_x);
        //solver.actual.solve(&mut f, &b_vec, false)?;
        Ok(Self { solver })
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
    b_mat: Option<&ComplexCooMatrix>,
    num_eigenvalues: usize,
    sigma: Sigma,
    genie: Genie,
) -> Result<ArpackResults, StrError> {
    let config = ArpackConfig {
        bmat: match &b_mat {
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

    let mut solver = ComplexLinSolver::new(genie)?;
    solver.actual.factorize(&a_mat, None)?;

    let mut driver = ArpackDriver::new(config, matrix_vector_product, linear_solve, solver);
    driver.solve()
}

/// Computes the matrix-vector product y = A*x
///
/// For this example, A is a tridiagonal matrix from the
/// discretization of a 1D convection-diffusion operator.
///
/// # Parameters:
/// * `x`: Input vector
/// * `y`: Output vector, will contain A*x
fn matrix_vector_product(state: &mut ComplexLinSolver<'_>, x: &[ArpackComplex64], y: &mut [ArpackComplex64]) {
    todo!("Implement matrix-vector multiplication");
}

/// Solves the linear system (A - sigma*I)*y = x
///
/// This is the core operation for shift-invert mode.
///
/// # Parameters:
/// * `x`: Input vector (right-hand side)
/// * `y`: Output vector (solution)
/// * `sigma`: Shift value
fn linear_solve(
    state: &mut ComplexLinSolver<'_>,
    x: &[ArpackComplex64],
    y: &mut [ArpackComplex64],
    sigma: ArpackComplex64,
) {
    todo!("Implement linear system solver");

    // Example implementation would:
    // 1. Form the shifted matrix (A - sigma*I)
    // 2. Factor this matrix (could cache this if sigma doesn't change)
    // 3. Solve the linear system
}
