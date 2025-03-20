use image::{GenericImageView, imageops::FilterType};
use ndarray::Array4;
use ort::session::{Session, builder::GraphOptimizationLevel};
// use show_image::{ImageInfo, ImageView, create_window, event};

// #[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the image
    let original_img =
        image::open("../data/lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")?;
    let (width, height) = original_img.dimensions();
    println!("height: {height}, width: {width}");

    // Convert to raw RGBA8 format
    // let img_u8 = original_img.to_rgba8().into_raw();

    // Convert model input
    let img_resized = original_img.resize_exact(112, 112, FilterType::Nearest);
    let img_f32 = img_resized.to_rgb32f().into_raw();
    // Convert to ndarray tensor shape (1, 3, 112, 112)
    let input_tensor = Array4::from_shape_vec((1, 3, 112, 112), img_f32)?;

    // Perform inference
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("../checkpoints/onnx/crfiqa-s.onnx")?;

    let outputs = model.run(ort::inputs!["input" => input_tensor]?)?;
    let predictions = outputs["qs"].try_extract_tensor::<f32>()?;

    // Extract float from output tensor
    // println!("{predictions}");
    // println!("Array shape: {:?}", predictions.shape());
    for entry in predictions.iter() {
        println!("qs: {:.4}", entry);
    }

    // // Equivalent to cv::imshow()
    // let image = ImageView::new(ImageInfo::rgba8(width, height), &img_u8);
    // let window = create_window("Image Viewer", Default::default())?;
    // window.set_image("image-001", image)?;

    // // Equivalent to cv::waitKey()
    // for event in window.event_channel()? {
    //     if let event::WindowEvent::KeyboardInput(event) = event {
    //         // println!("{:#?}", event);
    //         if event.input.key_code == Some(event::VirtualKeyCode::Escape)
    //             && event.input.state.is_pressed()
    //         {
    //             break;
    //         }
    //     }
    // }

    Ok(())
}
