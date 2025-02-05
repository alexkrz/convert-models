#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;

void show_progress(const int index, const int total)
{
    float progress = static_cast<float>(index) / total;
    int barWidth = 70; // Width of the progress bar
    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i)
    {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    // Display percentage and index/total
    std::cout << "] " << int(progress * 100.0) << "% "
              << "(" << index << "/" << total << ")\r";
    std::cout.flush();
}

void inference(
    const std::string& data_dir = "/home/kurz/git-work/convert-models/data/lfw",
    const std::string& data_name = "lfw",
    const std::string& file_ext = ".jpg",
    const std::string& method_name = "crfiqa-l",
    const std::string& checkpoint_fp = "/home/kurz/git-work/convert-models/checkpoints/onnx/crfiqa-l.onnx",
    const std::string& save_dir = "/home/kurz/git-work/convert-models/results")
{
    // Check directories
    fs::path data_path(data_dir);
    fs::path save_path(save_dir);
    if (!fs::exists(save_path))
    {
        fs::create_directory(save_path);
    }

    // Glob data_dir
    std::vector<fs::path> img_list;
    for (const auto& entry : fs::directory_iterator(data_path))
    {
        if (entry.path().extension() == file_ext)
        {
            img_list.push_back(entry.path());
        }
    }
    std::sort(img_list.begin(), img_list.end());

    // Read model and set input_size
    std::cout << "Loading ONNX model.." << std::endl;
    cv::dnn::Net model = cv::dnn::readNetFromONNX(checkpoint_fp);
    cv::Size input_size(112, 112);

    std::vector<std::string> filename_list;
    std::vector<float> qs_scores_arr(img_list.size(), 0.0f);

    for (size_t idx = 0; idx < img_list.size(); ++idx)
    {
        show_progress(idx, img_list.size());
        fs::path img_p = img_list[idx];
        filename_list.push_back(img_p.filename().string());

        // Prepare model input
        cv::Mat img = cv::imread(img_p.string());

        // Show image
        // cv::namedWindow("Display window", cv::WINDOW_NORMAL);
        // cv::imshow("Display window", img);
        // cv::waitKey(0);

        cv::Mat blob;
        cv::dnn::blobFromImage(img, blob, 1.0 / 127.5, input_size, cv::Scalar(127.5, 127.5, 127.5), true, false);

        // Run forward pass
        model.setInput(blob);
        // std::vector<cv::Mat> outputs;
        std::vector<cv::String> outLayerNames = model.getUnconnectedOutLayersNames(); // Why does it not return the "feats" node?
        cv::Mat model_output = model.forward("qs");
        float qs = model_output.at<float>(0, 0); // Would like to have bounds checking here
        qs_scores_arr[idx] = qs;

        if (idx == 199)
        {
            break;
        }
    }

    std::cout << "\nSaving results.." << std::endl;
    qs_scores_arr.resize(img_list.size());
    std::cout << "Shape of qs_scores_arr: " << qs_scores_arr.size() << std::endl;

    std::ofstream out_file(save_path / (method_name + "_" + data_name + "_onnx_cpp.txt"));
    out_file << std::fixed << std::setprecision(6);
    out_file << "sample quality\n";
    for (size_t i = 0; i < filename_list.size(); ++i)
    {
        out_file << filename_list[i] << " " << qs_scores_arr[i] << "\n";
    }
    out_file.close();
}

int main()
{
    inference();
    return 0;
}
