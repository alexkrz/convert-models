use std::process;

use rust::Config;

// #[show_image::main]
fn main() {
    let config = Config::new();

    // Optional: Display image
    // if let Err(e) = rust::display_image(&config.image_p) {
    //     eprintln!("Could not show image: {}", e);
    //     process::exit(1);
    // }

    match rust::model_inference(&config.model_p, &config.image_p) {
        Ok(result) => {
            for entry in result {
                println!("qs: {:.4}", entry);
            }
        }
        Err(e) => {
            eprintln!("Application error: {}", e);
            process::exit(1);
        }
    }
}
