use std::process;

use rust::Config;

fn main() {
    let config = Config::new();

    if let Err(e) = rust::run(config) {
        eprintln!("Application error: {}", e);
        process::exit(1);
    }
}
