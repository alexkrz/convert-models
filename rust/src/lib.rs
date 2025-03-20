use clap::Parser;
use image::{GenericImageView, imageops::FilterType};
use ndarray::Array4;
use ort::session::{Session, builder::GraphOptimizationLevel};
use show_image::{ImageInfo, ImageView, create_window, event};
use std::error::Error;

/// Configuration for the inference
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Config {
    /// Path to the model file
    #[arg(short, long)]
    pub model_p: String,

    /// Path to the image file
    #[arg(short, long)]
    pub image_p: String,
}

impl Config {
    pub fn new() -> Self {
        Config::parse()
    }
}

pub fn display_image(image_p: &str) -> Result<(), Box<dyn Error>> {
    // Load the image
    let original_img = image::open(image_p)?;
    let (width, height) = original_img.dimensions();
    println!("height: {height}, width: {width}");

    // Convert to raw RGBA8 format
    let img_u8 = original_img.to_rgba8().into_raw();

    // Equivalent to cv::imshow()
    let image = ImageView::new(ImageInfo::rgba8(width, height), &img_u8);
    let window = create_window("Image Viewer", Default::default())?;
    window.set_image("image-001", image)?;

    // Equivalent to cv::waitKey()
    for event in window.event_channel()? {
        if let event::WindowEvent::KeyboardInput(event) = event {
            // println!("{:#?}", event);
            if event.input.key_code == Some(event::VirtualKeyCode::Escape)
                && event.input.state.is_pressed()
            {
                break;
            }
        }
    }
    Ok(())
}

pub fn model_inference(model_p: &str, image_p: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut result = Vec::new();

    // Load the image
    let original_img = image::open(image_p)?;
    let (width, height) = original_img.dimensions();
    println!("height: {height}, width: {width}");

    // Convert model input
    let img_resized = original_img.resize_exact(112, 112, FilterType::Nearest);
    let img_f32 = img_resized.to_rgb32f().into_raw();
    // Convert to ndarray tensor shape (1, 3, 112, 112)
    let input_tensor = Array4::from_shape_vec((1, 3, 112, 112), img_f32)?;

    // Perform inference
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_p)?;

    let outputs = model.run(ort::inputs!["input" => input_tensor]?)?;
    let predictions = outputs["qs"].try_extract_tensor::<f32>()?;

    // Extract float from output tensor
    // println!("{predictions}");
    // println!("Array shape: {:?}", predictions.shape());
    for entry in predictions.iter() {
        result.push(*entry);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn crfiqa_s() {
        let model_p = "../checkpoints/onnx/crfiqa-s.onnx";
        let image_p = "../data/lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg";

        let result = model_inference(model_p, image_p).unwrap(); // Panics if Err
        assert_abs_diff_eq!(vec![1.4946].as_slice(), result.as_slice(), epsilon = 1e-4);
    }
}
