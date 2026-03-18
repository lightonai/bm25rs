fn main() {
    #[cfg(feature = "cuda")]
    {
        // Add standard CUDA library search paths for the linker.
        let candidates = [
            "/usr/local/cuda/lib64/stubs",
            "/usr/local/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu",
        ];
        for path in &candidates {
            if std::path::Path::new(path).exists() {
                println!("cargo:rustc-link-search=native={}", path);
            }
        }

        // Support custom CUDA_PATH
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            let lib = format!("{}/lib64/stubs", cuda_path);
            if std::path::Path::new(&lib).exists() {
                println!("cargo:rustc-link-search=native={}", lib);
            }
            let lib64 = format!("{}/lib64", cuda_path);
            if std::path::Path::new(&lib64).exists() {
                println!("cargo:rustc-link-search=native={}", lib64);
            }
        }

        // Glob for versioned cuda installs: /usr/local/cuda-12*/lib64/stubs
        if let Ok(entries) = std::fs::read_dir("/usr/local") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with("cuda-") {
                    let stubs = format!("/usr/local/{}/lib64/stubs", name);
                    if std::path::Path::new(&stubs).exists() {
                        println!("cargo:rustc-link-search=native={}", stubs);
                    }
                }
            }
        }
    }
}
