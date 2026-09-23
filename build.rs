//! Build script: bakes engine identity into the binary so the UCI `id` lines
//! and `--version` output never drift from `Cargo.toml`.

fn main() {
    println!(
        "cargo:rustc-env=MORSTILIA_NAME=Morstilia {}",
        env!("CARGO_PKG_VERSION")
    );
    println!("cargo:rustc-env=MORSTILIA_ID_NAME=Morstilia");
    println!("cargo:rustc-env=MORSTILIA_ID_AUTHOR=Alp Dumlupınar");
    // Re-run if the version changes.
    println!("cargo:rerun-if-changed=Cargo.toml");
}
