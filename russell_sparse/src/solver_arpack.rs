use russell_lab::Complex64;

use crate::{arpack_ffi, StrError};
use std::{
    ffi::{c_char, c_int},
    marker::PhantomData,
};

// Comments in this file have been taken from function defintions:
// https://github.com/opencollab/arpack-ng/blob/master/SRC/znaupd.f
// https://github.com/opencollab/arpack-ng/blob/master/SRC/zneupd.f
// More information is available in the original files.
//
// Array indices in comments have been changed from Fortran indices that start at 1
// to Rust indices that start at 0 but there might be some mistakes so be vigilant.
//
// Inspiration has also been taken from the examples:
// https://github.com/opencollab/arpack-ng/blob/master/EXAMPLES/COMPLEX/zndrv2.f
// https://github.com/opencollab/arpack-ng/blob/master/EXAMPLES/COMPLEX/zndrv4.f
// More information is available in the original files.

// TODO: Implement support for all precision and data types
// ssaupd: double precision real (symmetric)
// snaupd: double precision real (non-symmetric)
// dsaupd: single precision real (symmetric)
// dnaupd: single precision real (non-symmetric)
// cnaupd: single precision complex
// znaupd: double precision complex

pub type ArpackComplex64 = arpack_ffi::__BindgenComplex<f64>;

impl ArpackComplex64 {
    fn zero() -> Self {
        ArpackComplex64 { re: 0.0, im: 0.0 }
    }
}

impl From<Complex64> for ArpackComplex64 {
    fn from(value: Complex64) -> Self {
        ArpackComplex64 {
            re: value.re,
            im: value.im,
        }
    }
}

/// Which eigenvalues to compute.
#[derive(Debug, Clone, Copy)]
pub enum EigenvaluePosition {
    /// Want the NEV eigenvalues of largest magnitude.
    LargestMagnitude,
    /// Want the NEV eigenvalues of smallest magnitude.
    SmallestMagnitude,
    /// Want the NEV eigenvalues of largest real part.
    LargestReal,
    /// Want the NEV eigenvalues of smallest real part.
    SmallestReal,
    /// Want the NEV eigenvalues of largest imaginary part.
    LargestImaginary,
    /// Want the NEV eigenvalues of smallest imaginary part.
    SmallestImaginary,
}

impl EigenvaluePosition {
    fn cstr(&self) -> *const c_char {
        match self {
            EigenvaluePosition::LargestMagnitude => c"LM".as_ptr(),
            EigenvaluePosition::SmallestMagnitude => c"SM".as_ptr(),
            EigenvaluePosition::LargestReal => c"LR".as_ptr(),
            EigenvaluePosition::SmallestReal => c"SR".as_ptr(),
            EigenvaluePosition::LargestImaginary => c"LI".as_ptr(),
            EigenvaluePosition::SmallestImaginary => c"SI".as_ptr(),
        }
    }
}

/// The form of the eigenvalue problem to solve.
#[derive(Debug, Clone, Copy)]
pub enum EigenValueProblem {
    /// Standard eigenvalue problem A*x = lambda*x.
    Standard,
    /// Generalized eigenvalue problem A*x = lambda*M*x.
    Generalized,
}

impl EigenValueProblem {
    fn cstr(&self) -> *const c_char {
        match self {
            EigenValueProblem::Standard => c"I".as_ptr(),
            EigenValueProblem::Generalized => c"G".as_ptr(),
        }
    }
}

/// The solving mode that Arpack should use.
#[derive(Debug, Clone, Copy)]
pub enum ArpackMode {
    // A*x = lambda*x.
    // ===> OP = A  and  B = I.
    Mode1,

    // A*x = lambda*M*x, M hermitian positive definite
    // ===> OP = inv[M]*A  and  B = M.
    Mode2,

    // A*x = lambda*M*x, M hermitian semi-definite
    // ===> OP =  inv[A - sigma*M]*M   and  B = M.
    // ===> shift-and-invert mode
    // If OP*x = amu*x, then lambda = sigma + 1/amu.
    Mode3,
}

impl ArpackMode {
    fn cint(&self) -> c_int {
        match self {
            ArpackMode::Mode1 => 1,
            ArpackMode::Mode2 => 2,
            ArpackMode::Mode3 => 3,
        }
    }
}

/// Configuration parameters for the ARPACK solver
#[derive(Debug, Clone)]
pub struct ArpackConfig {
    /// BMAT specifies the type of the matrix B that defines the
    /// semi-inner product for the operator OP.
    pub bmat: EigenValueProblem,

    /// Dimension of the eigenproblem.
    pub n: c_int,

    /// Which eigenvalues to compute.
    pub which: EigenvaluePosition,

    /// Number of eigenvalues of OP to be computed. 0 < NEV < N-1. // FIXME: assert
    pub nev: c_int,

    /// Stopping criteria: the relative accuracy of the Ritz value
    /// is considered acceptable if BOUNDS(I) .LE. TOL*ABS(RITZ(I))
    /// where ABS(RITZ(I)) is the magnitude when RITZ(I) is complex.
    /// DEFAULT = dlamch ('EPS') (machine precision as computed
    /// by the LAPACK auxiliary subroutine dlamch ).
    pub tol: f64,

    /// maximum number of Arnoldi update iterations allowed.
    pub mxiter: c_int,

    /// Number of columns of the matrix V. NCV must satisfy the two
    /// inequalities 1 <= NCV-NEV and NCV <= N.
    /// This will indicate how many Arnoldi vectors are generated
    /// at each iteration. After the startup phase in which NEV
    /// Arnoldi vectors are generated, the algorithm generates
    /// approximately NCV-NEV Arnoldi vectors at each subsequent update
    /// iteration. Most of the cost in generating each Arnoldi vector is
    /// in the matrix-vector operation OP*x.
    pub ncv: c_int,

    /// The mode to use for solving.
    pub mode: ArpackMode,

    /// Shift value for shift-invert mode (not relevant in other modes).
    pub shift: ArpackComplex64,
}

/// Results returned by the ARPACK solver
pub struct ArpackResults {
    /// The computed eigenvalues
    pub eigenvalues: Vec<ArpackComplex64>,
    /// The computed eigenvectors (stored as columns in row-major order)
    pub eigenvectors: Vec<ArpackComplex64>,
    /// Number of converged eigenvalues
    pub num_converged: usize,
    /// Number of iterations performed
    pub iterations: usize,
    /// Number of matrix-vector operations performed
    pub operations_count: usize,
}

/// The driver is responsible for performing matrix operations on behalf of ARPACK
pub struct ArpackDriver<'a, S, F, G>
where
    F: FnMut(&mut S, &[ArpackComplex64], &mut [ArpackComplex64]) + 'a,
    G: FnMut(&mut S, &[ArpackComplex64], &mut [ArpackComplex64], ArpackComplex64) + 'a,
{
    /// Configuration parameters
    cfg: ArpackConfig,
    /// Function that performs matrix-vector multiplication: A*x.
    matrix_vector_product: F,
    /// Function that solves the linear system: (A-sigma*I)*y = x.
    linear_solve: G,
    /// The driver's internal state that it could use to store A for instance
    state: S,
    /// PhantomData to hold the lifetime
    _phantom: PhantomData<&'a ()>,
}

impl<'a, S, F, G> ArpackDriver<'a, S, F, G>
where
    F: FnMut(&mut S, &[ArpackComplex64], &mut [ArpackComplex64]) + 'a,
    G: FnMut(&mut S, &[ArpackComplex64], &mut [ArpackComplex64], ArpackComplex64) + 'a,
{
    /// Create a new solver for a standard eigenvalue problem using shift-invert mode
    pub fn new(cfg: ArpackConfig, matrix_vector_product: F, linear_solve: G, state: S) -> Self {
        Self {
            cfg,
            matrix_vector_product,
            linear_solve,
            state,
            _phantom: PhantomData,
        }
    }

    /// Solve the eigenvalue problem
    pub fn solve(&mut self) -> Result<ArpackResults, StrError> {
        // IDO     Integer.  (INPUT/OUTPUT)
        //
        // Reverse communication flag. IDO must be zero on the first
        // call to znaupd. IDO will be set internally to
        // indicate the type of operation to be performed. Control is
        // then given back to the calling routine which has the
        // responsibility to carry out the requested operation and call
        // znaupd with the result. The operand is given in
        // WORKD(IPNTR(0)), the result must be put in WORKD(IPNTR(1)).
        // -------------------------------------------------------------
        // IDO =  0: first call to the reverse communication interface
        // IDO = -1: compute  Y = OP * X  where
        //           IPNTR(0) is the pointer into WORKD for X,
        //           IPNTR(1) is the pointer into WORKD for Y.
        //           This is for the initialization phase to force the
        //           starting vector into the range of OP.
        // IDO =  1: compute  Y = OP * X  where
        //           IPNTR(0) is the pointer into WORKD for X,
        //           IPNTR(1) is the pointer into WORKD for Y.
        //           In mode 3, the vector B * X is already
        //           available in WORKD(ipntr(2)).  It does not
        //           need to be recomputed in forming OP * X.
        // IDO =  2: compute  Y = M * X  where
        //           IPNTR(0) is the pointer into WORKD for X,
        //           IPNTR(1) is the pointer into WORKD for Y.
        // IDO =  3: compute and return the shifts in the first
        //           NP locations of WORKL.
        // IDO = 99: done
        // -------------------------------------------------------------
        // After the initialization phase, when the routine is used in
        // the "shift-and-invert" mode, the vector M * X is already
        // available and does not need to be recomputed in forming OP*X.
        let mut ido = 0;

        let bmat = self.cfg.bmat.cstr();
        let n = self.cfg.n as usize;
        let which = self.cfg.which.cstr();
        let nev = self.cfg.nev as usize;
        let tol = self.cfg.tol;

        // Initial residual vector and work array
        //      On input (if info!=0): initial residual vector
        //      On output: final residual vector
        let mut resid = vec![ArpackComplex64::zero(); n];

        let ncv = self.cfg.ncv as usize;

        // V       Complex*16  array N by NCV.  (OUTPUT)
        //
        // Contains the final set of Arnoldi basis vectors.
        let mut v = vec![ArpackComplex64::zero(); n * ncv];

        // LDV     Integer.  (INPUT)
        // Leading dimension of V exactly as declared in the calling program.
        let ldv = n; // FIXME: sure?

        // IPARAM  Integer array of length 11.  (INPUT/OUTPUT)
        //
        // IPARAM(0) = ISHIFT: method for selecting the implicit shifts.
        // The shifts selected at each iteration are used to filter out
        // the components of the unwanted eigenvector.
        // -------------------------------------------------------------
        // ISHIFT = 0: the shifts are to be provided by the user via
        //             reverse communication.  The NCV eigenvalues of
        //             the Hessenberg matrix H are returned in the part
        //             of WORKL array corresponding to RITZ.
        // ISHIFT = 1: exact shifts with respect to the current
        //             Hessenberg matrix H.  This is equivalent to
        //             restarting the iteration from the beginning
        //             after updating the starting vector with a linear
        //             combination of Ritz vectors associated with the
        //             "wanted" eigenvalues.
        // ISHIFT = 2: other choice of internal shift to be defined.
        // -------------------------------------------------------------
        //
        // IPARAM(1) = No longer referenced
        //
        // IPARAM(2) = MXITER
        // On INPUT:  maximum number of Arnoldi update iterations allowed.
        // On OUTPUT: actual number of Arnoldi update iterations taken.
        //
        // IPARAM(3) = NB: blocksize to be used in the recurrence.
        // The code currently works only for NB = 1.
        //
        // IPARAM(4) = NCONV: number of "converged" Ritz values.
        // This represents the number of Ritz values that satisfy
        // the convergence criterion.
        //
        // IPARAM(5) = IUPD
        // No longer referenced. Implicit restarting is ALWAYS used.
        //
        // IPARAM(6) = MODE
        // On INPUT determines what type of eigenproblem is being solved.
        // Must be 1,2,3.
        //
        // IPARAM(7) = NP
        // When ido = 3 and the user provides shifts through reverse
        // communication (IPARAM(1)=0), _naupd returns NP, the number
        // of shifts the user is to provide. 0 < NP < NCV-NEV.
        //
        // IPARAM(8) = NUMOP, IPARAM(9) = NUMOPB, IPARAM(10) = NUMREO,
        // OUTPUT: NUMOP  = total number of OP*x operations,
        //         NUMOPB = total number of B*x operations if BMAT='G',
        //         NUMREO = total number of steps of re-orthogonalization.
        let mut iparam = [0; 11];
        iparam[0] = 1; // Exact shift is recommended
        iparam[2] = self.cfg.mxiter;
        iparam[3] = 1;
        iparam[6] = self.cfg.mode.cint();

        // IPNTR   Integer array of length 14.  (OUTPUT)
        //
        // Pointer to mark the starting locations in the WORKD and WORKL
        // arrays for matrices/vectors used by the Arnoldi iteration.
        // -------------------------------------------------------------
        // IPNTR(0): pointer to the current operand vector X in WORKD.
        // IPNTR(1): pointer to the current result vector Y in WORKD.
        // IPNTR(2): pointer to the vector B * X in WORKD when used in
        //           the shift-and-invert mode.
        // IPNTR(3): pointer to the next available location in WORKL
        //           that is untouched by the program.
        // IPNTR(4): pointer to the NCV by NCV upper Hessenberg
        //           matrix H in WORKL.
        // IPNTR(5): pointer to the  ritz value array  RITZ
        // IPNTR(6): pointer to the (projected) ritz vector array Q
        // IPNTR(7): pointer to the error BOUNDS array in WORKL.
        // IPNTR(13): pointer to the NP shifts in WORKL.
        //
        // Note: IPNTR(8:12) is only referenced by zneupd.
        //
        // IPNTR(8): pointer to the NCV RITZ values of the
        //           original system.
        // IPNTR(9): Not Used
        // IPNTR(10): pointer to the NCV corresponding error bounds.
        // IPNTR(11): pointer to the NCV by NCV upper triangular
        //            Schur matrix for H.
        // IPNTR(12): pointer to the NCV by NCV matrix of eigenvectors
        //            of the upper Hessenberg matrix H. Only referenced by
        //            zneupd  if RVEC = .TRUE.
        let mut ipntr = [0; 14];

        // WORKD   Complex*16  work array of length 3*N.  (REVERSE COMMUNICATION)
        //
        // Distributed array to be used in the basic Arnoldi iteration
        // for reverse communication. The user should not use WORKD
        // as temporary workspace during the iteration !!!!!!!!!!
        let mut workd = vec![ArpackComplex64::zero(); 3 * n];

        // LWORKL  Integer.  (INPUT)
        //
        // LWORKL must be at least 3*NCV**2 + 5*NCV.
        let lworkl = 3 * ncv * ncv + 5 * ncv;

        // WORKL   Complex*16  work array of length LWORKL.  (OUTPUT/WORKSPACE)
        //
        // Private (replicated) array on each PE or array allocated on the front end.
        let mut workl = vec![ArpackComplex64::zero(); lworkl];

        // RWORK   Double precision   work array of length NCV (WORKSPACE)
        //
        // Private (replicated) array on each PE or array allocated on the front end.
        let mut rwork = vec![0.0; ncv];

        // INFO    Integer.  (INPUT/OUTPUT)
        //
        // If INFO .EQ. 0, a randomly initial residual vector is used.
        // If INFO .NE. 0, RESID contains the initial residual vector, possibly from a previous run.
        //
        // Error flag on output.
        // =  0: Normal exit.
        // =  1: Maximum number of iterations taken.
        //       All possible eigenvalues of OP has been found. IPARAM(5)
        //       returns the number of wanted converged Ritz values.
        // =  2: No longer an informational error. Deprecated starting
        //       with release 2 of ARPACK.
        // =  3: No shifts could be applied during a cycle of the
        //       Implicitly restarted Arnoldi iteration. One possibility
        //       is to increase the size of NCV relative to NEV.
        //       See remark 4 below.
        // = -1: N must be positive.
        // = -2: NEV must be positive.
        // = -3: NCV-NEV >= 2 and less than or equal to N.
        // = -4: The maximum number of Arnoldi update iteration
        //       must be greater than zero.
        // = -5: WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'
        // = -6: BMAT must be one of 'I' or 'G'.
        // = -7: Length of private work array is not sufficient.
        // = -8: Error return from LAPACK eigenvalue calculation;
        // = -9: Starting vector is zero.
        // = -10: IPARAM(6) must be 1,2,3.
        // = -11: IPARAM(6) = 1 and BMAT = 'G' are incompatible.
        // = -12: IPARAM(0) must be equal to 0 or 1.
        // = -9999: Could not build an Arnoldi factorization.
        //          User input error highly likely.  Please
        //          check actual array dimensions and layout.
        //          IPARAM(4) returns the size of the current Arnoldi
        //          factorization.
        let mut info = 0;

        // ----------------------------------------------
        // Reverse communication loop
        // ----------------------------------------------

        loop {
            unsafe {
                // ARPACK reverse communication interface routine.
                arpack_ffi::znaupd_c(
                    &mut ido,
                    bmat,
                    n as i32,
                    which,
                    nev as i32,
                    tol,
                    resid.as_mut_ptr(),
                    ncv as i32,
                    v.as_mut_ptr(),
                    ldv as i32,
                    iparam.as_mut_ptr(),
                    ipntr.as_mut_ptr(),
                    workd.as_mut_ptr(),
                    workl.as_mut_ptr(),
                    lworkl as i32,
                    rwork.as_mut_ptr(),
                    &mut info,
                );
            }

            if info < 0 {
                // FIXME: return proper error message

                eprintln!("Error in ARPACK znaupd: {}", info);
                return Err("Error in ARPACK znaupd");
            }

            match ido {
                -1 => {
                    // Perform y <--- OP*x = inv[A-SIGMA*M]*M*x
                    // to force starting vector into the range
                    // of OP. The user should supply his/her
                    // own matrix vector multiplication routine
                    // and a linear system solver. The matrix
                    // vector multiplication routine should take
                    // workd(ipntr(0)) as the input. The final
                    // result should be returned to workd(ipntr(1)).

                    // We need to account for Fortran indices starting at 1 instead of 0
                    let in_start = (ipntr[0] - 1) as usize;
                    let out_start = (ipntr[1] - 1) as usize;

                    let in_stop = in_start + (n as usize);
                    let out_stop = out_start + (n as usize);

                    let (input, output) = {
                        if out_start > in_stop {
                            let (a, b) = workd.split_at_mut(out_start);
                            let input = &a[in_start..in_stop];
                            let output = &mut b[..(n as usize)];
                            (input, output)
                        } else {
                            let (a, b) = workd.split_at_mut(in_start);
                            let input = &b[..(n as usize)];
                            let output = &mut a[out_start..out_stop];
                            (input, output)
                        }
                    };

                    (self.linear_solve)(&mut self.state, input, output, self.cfg.shift);
                    // FIXME
                }
                1 => {
                    // Perform y <-- OP*x = inv[A-sigma*M]*M*x
                    // M*x has been saved in workd(ipntr(2)).
                    // The user only need the linear system
                    // solver here that takes workd(ipntr(2))
                    // as input, and returns the result to
                    // workd(ipntr(1)).

                    // We need to account for Fortran indices starting at 1 instead of 0
                    let in_start = (ipntr[2] - 1) as usize;
                    let out_start = (ipntr[1] - 1) as usize;

                    let in_stop = in_start + (n as usize);
                    let out_stop = out_start + (n as usize);

                    let (input, output) = {
                        if out_start > in_stop {
                            let (a, b) = workd.split_at_mut(out_start);
                            let input = &a[in_start..in_stop];
                            let output = &mut b[..(n as usize)];
                            (input, output)
                        } else {
                            let (a, b) = workd.split_at_mut(in_start);
                            let input = &b[..(n as usize)];
                            let output = &mut a[out_start..out_stop];
                            (input, output)
                        }
                    };

                    (self.linear_solve)(&mut self.state, input, output, self.cfg.shift);
                    // FIXME
                }
                2 => todo!(),
                3 => unimplemented!(),
                99 => break, // Computation completed or error
                _ => {
                    return {
                        eprintln!("Unexpected IDO value: {}", ido);
                        Err("Unexpected IDO value")
                    }
                }
            }
        }

        // ----------------------------------------------
        // Extract eigenvalues and eigenvectors
        // ----------------------------------------------

        // RVEC    LOGICAL  (INPUT)
        //
        // Specifies whether a basis for the invariant subspace corresponding
        // to the converged Ritz value approximations for the eigenproblem
        // A*z = lambda*B*z is computed.
        //
        //    RVEC = .FALSE.     Compute Ritz values only.
        //
        //    RVEC = .TRUE.      Compute Ritz vectors or Schur vectors.
        //                       See Remarks below.
        let rvec = 1;

        // HOWMNY  Character*1  (INPUT)
        //
        // Specifies the form of the basis for the invariant subspace
        // corresponding to the converged Ritz values that is to be computed.
        //
        // = 'A': Compute NEV Ritz vectors;
        // = 'P': Compute NEV Schur vectors;
        // = 'S': compute some of the Ritz vectors, specified
        //        by the logical array SELECT.
        let howmny = c"A".as_ptr();

        //SELECT  Logical array of dimension NCV.  (INPUT)
        //
        // If HOWMNY = 'S', SELECT specifies the Ritz vectors to be
        // computed. To select the  Ritz vector corresponding to a
        // Ritz value D(j), SELECT(j) must be set to .TRUE..
        // If HOWMNY = 'A' or 'P', SELECT need not be initialized
        // but it is used as internal workspace.
        let select = vec![0; ncv];

        // Complex*16 array of dimension NEV+1.  (OUTPUT)
        //
        // On exit, D contains the  Ritz  approximations
        // to the eigenvalues lambda for A*z = lambda*B*z.
        let mut d = vec![ArpackComplex64::zero(); nev + 1];

        // Z       Complex*16 N by NEV array    (OUTPUT)
        //
        // On exit, if RVEC = .TRUE. and HOWMNY = 'A', then the columns of
        // Z represents approximate eigenvectors (Ritz vectors) corresponding
        // to the NCONV=IPARAM(5) Ritz values for eigensystem
        // A*z = lambda*B*z.
        //
        // If RVEC = .FALSE. or HOWMNY = 'P', then Z is NOT REFERENCED.
        //
        // NOTE: If if RVEC = .TRUE. and a Schur basis is not required,
        // the array Z may be set equal to first NEV+1 columns of the Arnoldi
        // basis array V computed by ZNAUPD.  In this case the Arnoldi basis
        // will be destroyed and overwritten with the eigenvector basis.
        let mut z = vec![ArpackComplex64::zero(); n * nev];

        // LDZ     Integer.  (INPUT)
        //
        // The leading dimension of the array Z. If Ritz vectors are
        // desired, then  LDZ .ge.  max( 1, N ) is required.
        // In any case,  LDZ .ge. 1 is required.
        let ldz = n;

        // SIGMA   Complex*16  (INPUT)
        //
        // If IPARAM(7) = 3 then SIGMA represents the shift.
        // Not referenced if IPARAM(7) = 1 or 2.
        let sigma = self.cfg.shift;

        // WORKEV  Complex*16 work array of dimension 2*NCV.  (WORKSPACE)
        let mut workev = vec![ArpackComplex64::zero(); 2 * ncv];

        // The remaining arguments come from the call to znaupd_c.
        // Some arrays will now contain output values though.

        //  V       Complex*16 N by NCV array.  (INPUT/OUTPUT)
        //
        //  Upon INPUT: the NCV columns of V contain the Arnoldi basis
        //              vectors for OP as constructed by ZNAUPD.
        //
        //  Upon OUTPUT: If RVEC = .TRUE. the first NCONV=IPARAM(4) columns
        //               contain approximate Schur vectors that span the
        //               desired invariant subspace.
        //
        //  NOTE: If the array Z has been set equal to first NEV+1 columns
        //  of the array V and RVEC=.TRUE. and HOWMNY= 'A', then the
        //  Arnoldi basis held by V has been overwritten by the desired
        //  Ritz vectors.  If a separate array Z has been passed then
        //  the first NCONV=IPARAM(4) columns of V will contain approximate
        //  Schur vectors that span the desired invariant subspace.

        //  WORKL   Double precision work array of length LWORKL.  (OUTPUT/WORKSPACE)
        //
        //  WORKL(1:ncv*ncv+2*ncv) contains information obtained in
        //  znaupd.  They are not changed by zneupd.
        //  WORKL(ncv*ncv+2*ncv+1:3*ncv*ncv+4*ncv) holds the
        //  untransformed Ritz values, the untransformed error estimates of
        //  the Ritz values, the upper triangular matrix for H, and the
        //  associated matrix representation of the invariant subspace for H.
        //
        //  Note: IPNTR(8:12) contains the pointer into WORKL for addresses
        //  of the above information computed by zneupd.
        //  -------------------------------------------------------------
        //  IPNTR(8):  pointer to the NCV RITZ values of the
        //             original system.
        //  IPNTR(9): Not used
        //  IPNTR(10): pointer to the NCV corresponding error estimates.
        //  IPNTR(11): pointer to the NCV by NCV upper triangular
        //             Schur matrix for H.
        //  IPNTR(12): pointer to the NCV by NCV matrix of eigenvectors
        //             of the upper Hessenberg matrix H. Only referenced by
        //             zneupd if RVEC = .TRUE.
        //  -------------------------------------------------------------

        //  INFO    Integer.  (OUTPUT)
        //
        //  Error flag on output.
        //  =  0: Normal exit.
        //
        //  =  1: The Schur form computed by LAPACK routine csheqr
        //        could not be reordered by LAPACK routine ztrsen.
        //        Re-enter subroutine zneupd with IPARAM(4)=NCV and
        //        increase the size of the array D to have
        //        dimension at least dimension NCV and allocate at least NCV
        //        columns for Z. NOTE: Not necessary if Z and V share
        //        the same space. Please notify the authors if this error
        //        occurs.
        //
        //  = -1: N must be positive.
        //  = -2: NEV must be positive.
        //  = -3: NCV-NEV >= 1 and less than or equal to N.
        //  = -5: WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'
        //  = -6: BMAT must be one of 'I' or 'G'.
        //  = -7: Length of private work WORKL array is not sufficient.
        //  = -8: Error return from LAPACK eigenvalue calculation.
        //        This should never happened.
        //  = -9: Error return from calculation of eigenvectors.
        //        Informational error from LAPACK routine ztrevc.
        //  = -10: IPARAM(6) must be 1,2,3
        //  = -11: IPARAM(6) = 1 and BMAT = 'G' are incompatible.
        //  = -12: HOWMNY = 'S' not yet implemented
        //  = -13: HOWMNY must be one of 'A' or 'P' if RVEC = .true.
        //  = -14: ZNAUPD did not find any eigenvalues to sufficient
        //         accuracy.
        //  = -15: ZNEUPD got a different count of the number of converged
        //         Ritz values than ZNAUPD got. This indicates the user
        //         probably made an error in passing data from ZNAUPD to
        //         ZNEUPD or that the data was modified before entering
        //         ZNEUPD

        unsafe {
            // ARPACK routine that returns Ritz values and (optionally) Ritz vectors.
            arpack_ffi::zneupd_c(
                rvec,
                howmny,
                select.as_ptr(),
                d.as_mut_ptr(),
                z.as_mut_ptr(),
                ldz as i32,
                sigma,
                workev.as_mut_ptr(),
                bmat,
                n as i32,
                which,
                nev as i32,
                tol,
                resid.as_mut_ptr(),
                ncv as i32,
                v.as_mut_ptr(),
                n as i32,
                iparam.as_mut_ptr(),
                ipntr.as_mut_ptr(),
                workd.as_mut_ptr(),
                workl.as_mut_ptr(),
                lworkl as i32,
                rwork.as_mut_ptr(),
                &mut info,
            );
        }

        if info != 0 {
            eprintln!("Error in ARPACK zneupd: {}", info);
            return Err("Error in ARPACK zneupd");
        }

        Ok(ArpackResults {
            eigenvalues: d,
            eigenvectors: v,
            num_converged: iparam[4] as usize,
            iterations: iparam[2] as usize,
            operations_count: iparam[8] as usize,
        })
    }
}
