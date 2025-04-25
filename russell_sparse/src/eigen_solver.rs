use crate::arpack;
use std::{
    ffi::{c_int, CStr},
    marker::PhantomData,
};

type ArComplex = arpack::__BindgenComplex<f64>;

/// Which eigenvalues to compute
#[derive(Debug, Clone, Copy)]
pub enum EigenvaluePosition {
    /// Eigenvalues with largest magnitude
    LargestMagnitude,
    /// Eigenvalues with smallest magnitude
    SmallestMagnitude,
    /// Eigenvalues with largest real part
    LargestReal,
    /// Eigenvalues with smallest real part
    SmallestReal,
    /// Eigenvalues with largest imaginary part
    LargestImaginary,
    /// Eigenvalues with smallest imaginary part
    SmallestImaginary,
}

impl EigenvaluePosition {
    fn as_cstr(&self) -> &CStr {
        match self {
            EigenvaluePosition::LargestMagnitude => c"LM",
            EigenvaluePosition::SmallestMagnitude => c"SM",
            EigenvaluePosition::LargestReal => c"LR",
            EigenvaluePosition::SmallestReal => c"SR",
            EigenvaluePosition::LargestImaginary => c"LI",
            EigenvaluePosition::SmallestImaginary => c"SI",
        }
    }
}

/// Configuration parameters for the ARPACK solver
#[derive(Debug, Clone)]
pub struct ArpackConfig {
    /// Number of eigenvalues to compute
    pub num_eigenvalues: usize,
    /// Maximum number of Arnoldi iterations allowed
    pub max_iterations: usize,
    /// Number of Arnoldi vectors to use (usually 2*num_eigenvalues or more)
    pub num_arnoldi_vectors: usize,
    /// Which eigenvalues to compute
    pub which_eigenvalues: EigenvaluePosition,
    /// Convergence tolerance (use machine precision if set to 0.0)
    pub tolerance: f64,
    /// Shift value for shift-invert mode
    pub shift: ArComplex,
}

impl Default for ArpackConfig {
    fn default() -> Self {
        Self {
            num_eigenvalues: 6,
            max_iterations: 1000,
            num_arnoldi_vectors: 20,
            which_eigenvalues: EigenvaluePosition::LargestMagnitude,
            tolerance: 0.0,
            shift: ArComplex { re: 0.0, im: 0.0 },
        }
    }
}

/// Results returned by the ARPACK solver
pub struct EigenResults {
    /// The computed eigenvalues
    pub eigenvalues: Vec<ArComplex>,
    /// The computed eigenvectors (stored as columns in row-major order)
    pub eigenvectors: Vec<ArComplex>,
    /// Number of converged eigenvalues
    pub num_converged: usize,
    /// Number of iterations performed
    pub iterations: usize,
    /// Number of matrix-vector operations performed
    pub operations_count: usize,
}

/// The main ARPACK solver for shift-invert mode
pub struct ShiftInvertSolver<'a, F, G>
where
    F: FnMut(&[ArComplex], &mut [ArComplex]) + 'a,
    G: FnMut(&[ArComplex], &mut [ArComplex], ArComplex) + 'a,
{
    /// Dimension of the matrix
    dimension: usize,
    /// Configuration parameters
    config: ArpackConfig,
    /// Function that performs matrix-vector multiplication: A*x
    matrix_vector_product: F,
    /// Function that solves the linear system: (A-sigma*I)*y = x
    linear_solver: G,
    /// PhantomData to hold the lifetime
    _phantom: PhantomData<&'a ()>,
}

impl<'a, F, G> ShiftInvertSolver<'a, F, G>
where
    F: FnMut(&[ArComplex], &mut [ArComplex]) + 'a,
    G: FnMut(&[ArComplex], &mut [ArComplex], ArComplex) + 'a,
{
    /// Create a new solver for a standard eigenvalue problem using shift-invert mode
    pub fn new(dimension: usize, config: ArpackConfig, matrix_vector_product: F, linear_solver: G) -> Self {
        // Validate the configuration
        assert!(dimension > 0, "Matrix dimension must be positive");
        assert!(
            config.num_eigenvalues < dimension,
            "Number of requested eigenvalues must be less than matrix dimension"
        );
        assert!(
            config.num_arnoldi_vectors > config.num_eigenvalues + 1,
            "Number of Arnoldi vectors must be greater than num_eigenvalues + 1"
        );
        assert!(
            config.num_arnoldi_vectors <= dimension,
            "Number of Arnoldi vectors cannot exceed matrix dimension"
        );

        Self {
            dimension,
            config,
            matrix_vector_product,
            linear_solver,
            _phantom: PhantomData,
        }
    }

    /// Solve the eigenvalue problem
    pub fn solve(&mut self) -> Result<EigenResults, String> {
        let n = self.dimension as c_int;
        let nev = self.config.num_eigenvalues as c_int;
        let ncv = self.config.num_arnoldi_vectors as c_int;
        let max_iter = self.config.max_iterations as c_int;
        let sigma = self.config.shift;

        // -------------------------
        // ARPACK parameters
        // -------------------------

        // Reverse communication parameter
        //      0   = first call (ARPACK initializes everything)
        //      -1  = compute y = OP*x where OP = inv[A-sigma*I] and x is at workd[ipntr[0]-1]
        //      1   = compute y = OP*x where x is at workd[ipntr[0]-1]
        //      99  = computation completed or error
        let mut ido = 0;

        // Type of eigenvalue problem
        //      "I" = standard eigenvalue problem: A*x = lambda*x
        //      "G" = generalized eigenvalue problem: A*x = lambda*B*x
        let bmat = c"I".as_ptr();

        // Which eigenvalues to compute
        //      "LM" = eigenvalues with largest magnitude
        //      "SM" = eigenvalues with smallest magnitude
        //      "LR" = eigenvalues with largest real part
        //      "SR" = eigenvalues with smallest real part
        //      "LI" = eigenvalues with largest imaginary part
        //      "SI" = eigenvalues with smallest imaginary part
        let which = self.config.which_eigenvalues.as_cstr().as_ptr();

        // Stopping criteria - relative accuracy of Ritz values
        //      If ≤ 0, machine precision is used
        let tol = self.config.tolerance;

        // Input/output status flag
        //      On input: 0 = use random initial vector
        //               ≠0 = resid contains initial residual vector
        //      On output: 0 = normal exit
        //                >0 = number of iterations taken
        //                <0 = error code
        let mut info = 0;

        // -------------------------
        // ARPACK arrays
        // -------------------------

        // Initial residual vector and work array
        //      On input (if info!=0): initial residual vector
        //      On output: final residual vector
        let mut resid = vec![ArComplex { re: 0.0, im: 0.0 }; self.dimension];

        // Contains the Arnoldi basis vectors
        //    Stored as n x ncv matrix in column-major order
        //    After zneupd, first nconv columns contain eigenvectors
        let mut v = vec![ArComplex { re: 0.0, im: 0.0 }; self.dimension * self.config.num_arnoldi_vectors];

        // Working array used for reverse communication
        //      Input vectors and output vectors are stored here
        //      Must be at least 3*n in size
        //      The location of current input/output vectors indicated by ipntr
        let mut workd = vec![ArComplex { re: 0.0, im: 0.0 }; 3 * self.dimension];

        // Working array for ARPACK internal computations
        //      Must be at least 3*ncv^2 + 5*ncv in size
        let lworkl = 3 * ncv * ncv + 5 * ncv;
        let mut workl = vec![ArComplex { re: 0.0, im: 0.0 }; lworkl as usize];

        // Real work array for complex arithmetic calculations
        //      Must be at least ncv in size
        let mut rwork = vec![0.0; self.dimension];

        // Integer array of pointers (contains Fortran indices that start at 1)
        //      ipntr[0] = pointer to the current input vector in workd
        //      ipntr[1] = pointer to the current output vector in workd
        //      ipntr[2] = pointer to the next available location in workd
        //      (other values used internally by ARPACK)
        let mut ipntr = [0; 14];

        // Integer array of parameters
        //      iparam[0] = shift strategy (1=exact shifts)
        //      iparam[2] = max number of iterations
        //      iparam[4] = (output) number of converged eigenvalues
        //      iparam[6] = mode:
        //                 1 = A*x
        //                 2 = OP*x = inv[B]*A*x
        //                 3 = OP*x = inv[A-sigma*I]*x    (shift-invert mode)
        //                 4 = OP*x = inv[B-sigma*A]*A*x
        //                 5 = OP*x = inv[A-sigma*B]*B*x
        //      iparam[8] = (output) number of OP*x operations
        let mut iparam = [0; 11];
        iparam[0] = 1;
        iparam[2] = max_iter;
        iparam[6] = 3; // FIXME: Support more modes!

        // ----------------------------------------------
        // Main ARPACK reverse communication loop
        // ----------------------------------------------

        loop {
            unsafe {
                arpack::znaupd_c(
                    &mut ido,
                    bmat,
                    n,
                    which,
                    nev,
                    tol,
                    resid.as_mut_ptr(),
                    ncv,
                    v.as_mut_ptr(),
                    n,
                    iparam.as_mut_ptr(),
                    ipntr.as_mut_ptr(),
                    workd.as_mut_ptr(),
                    workl.as_mut_ptr(),
                    lworkl,
                    rwork.as_mut_ptr(),
                    &mut info,
                );
            }

            if info < 0 {
                return Err(format!("Error in ARPACK znaupd: {}", info));
            }

            // Handle reverse communication
            match ido {
                // FIXME: handle -1 and 1 differently
                -1 | 1 => {
                    // Compute y = inv[A-sigma*I]*x

                    // Fortran indices start at 1 instead of 0
                    let start_in = (ipntr[0] - 1) as usize;
                    let start_out = (ipntr[1] - 1) as usize;

                    let stop_in = start_in + self.dimension;
                    let stop_out = start_out + self.dimension;

                    let (input, output) = {
                        if start_out > stop_in {
                            let (a, b) = workd.split_at_mut(start_out);
                            let input = &a[start_in..stop_in];
                            let output = &mut b[..self.dimension];
                            (input, output)
                        } else {
                            let (a, b) = workd.split_at_mut(start_in);
                            let input = &b[..self.dimension];
                            let output = &mut a[start_out..stop_out];
                            (input, output)
                        }
                    };

                    // Call the user-provided linear solver
                    (self.linear_solver)(input, output, sigma);
                }
                99 => break, // Computation completed or error
                _ => return Err(format!("Unexpected IDO value: {}", ido)),
            }
        }

        // ----------------------------------------------
        // Extract eigenvalues and eigenvectors
        // ----------------------------------------------

        // rvec: Compute eigenvectors (1) or not (0)
        let rvec = 1;

        // How many eigenvectors to compute
        //      "A" = All eigenvectors
        //      "S" = Selected eigenvectors (using select array)
        let howmny = c"A".as_ptr();

        // Selection array for zneupd
        //      Specifies which Ritz values to use in zneupd
        //      Typically not set by user - zneupd handles it internally
        let select = vec![0; ncv as usize];

        // Array that will hold computed eigenvalues
        //    Size must be at least nev+1
        let mut d = vec![ArComplex { re: 0.0, im: 0.0 }; (nev + 1) as usize];

        // Work array for zneupd
        //      Must be 2*ncv in size
        let mut workev = vec![ArComplex { re: 0.0, im: 0.0 }; 2 * ncv as usize];

        unsafe {
            arpack::zneupd_c(
                rvec,
                howmny,
                select.as_ptr(),
                d.as_mut_ptr(),
                v.as_mut_ptr(), // Reuse v to store eigenvectors
                n,
                sigma,
                workev.as_mut_ptr(),
                bmat,
                n,
                which,
                nev,
                tol,
                resid.as_mut_ptr(),
                ncv,
                v.as_mut_ptr(),
                n,
                iparam.as_mut_ptr(),
                ipntr.as_mut_ptr(),
                workd.as_mut_ptr(),
                workl.as_mut_ptr(),
                lworkl,
                rwork.as_mut_ptr(),
                &mut info,
            );
        }

        if info != 0 {
            return Err(format!("Error in ARPACK zneupd: {}", info));
        }

        Ok(EigenResults {
            eigenvalues: d,
            eigenvectors: v,
            num_converged: iparam[4] as usize,
            iterations: iparam[2] as usize,
            operations_count: iparam[8] as usize,
        })
    }
}
