use image::GenericImageView;
use show_image::{ImageInfo, ImageView, create_window, event};

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the image
    let img = image::open("../data/lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")?;
    let (width, height) = img.dimensions();
    println!("height: {height}, width: {width}");

    // Convert to raw RGBA8 format
    let img = img.into_rgba8();
    let pixels = img.into_raw();

    // Equivalent to cv::imshow()
    let image = ImageView::new(ImageInfo::rgba8(width, height), &pixels);
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
