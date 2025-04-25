use std::{env, path::PathBuf};

fn main() {
    // SuiteSparse ----------------------------------------------------------
    let libs = vec!["klu", "umfpack"];

    #[cfg(not(feature = "local_suitesparse"))]
    let lib_dirs = vec![
        "/usr/lib/x86_64-linux-gnu", // Debian
        "/usr/lib",                  // Arch
        "/usr/lib64",                // Rocky
        "/opt/homebrew/lib",         // macOS
    ];

    #[cfg(not(feature = "local_suitesparse"))]
    let inc_dirs = vec![
        "/usr/include/suitesparse", // Linux
    ];

    #[cfg(target_os = "macos")]
    let inc_dirs = vec![
        "/opt/homebrew/include/suitesparse", // macOS
        "/usr/local/include/suitesparse",    // macOS
    ];

    #[cfg(feature = "local_suitesparse")]
    let lib_dirs = vec!["/usr/local/lib/suitesparse"];

    #[cfg(feature = "local_suitesparse")]
    let inc_dirs = vec!["/usr/local/include/suitesparse"];

    cc::Build::new()
        .file("c_code/interface_complex_klu.c")
        .file("c_code/interface_complex_umfpack.c")
        .file("c_code/interface_klu.c")
        .file("c_code/interface_umfpack.c")
        .includes(&inc_dirs)
        .compile("c_code_suitesparse");
    for d in &lib_dirs {
        println!("cargo:rustc-link-search=native={}", *d);
    }
    for l in &libs {
        println!("cargo:rustc-link-lib=dylib={}", *l);
    }

    // MUMPS ----------------------------------------------------------------

    #[cfg(feature = "with_mumps")]
    {
        cc::Build::new()
            .file("c_code/interface_complex_mumps.c")
            .file("c_code/interface_mumps.c")
            .include("/usr/local/include/mumps")
            .compile("c_code_mumps");
        println!("cargo:rustc-link-search=native=/usr/local/lib/mumps");
        println!("cargo:rustc-link-lib=dylib=dmumps_cpmech");
        println!("cargo:rustc-link-lib=dylib=zmumps_cpmech");
    }

    // ARPACK --------------------------------------------------------------
    let bindings = bindgen::Builder::default()
        .header("c_code/arpack.h")
        .generate()
        .expect("Unable to generate ARPACK bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("arpack_bindings.rs"))
        .expect("Couldn't write ARPACK bindings!");

    println!("cargo:rustc-link-lib=dylib=arpack");
}
