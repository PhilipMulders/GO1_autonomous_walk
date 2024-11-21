#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    // Chessboard size (number of internal corners)
    cv::Size boardSize(7, 7); // 8x8 chessboard pattern
    float squareSize = 2.0f; // cm

    // Arrays to store the object points (3D) and image points (2D)
    std::vector<std::vector<cv::Point3f>> objectPoints; 
    std::vector<std::vector<cv::Point2f>> imagePoints;

    // Generate 3D points for the chessboard (assuming it's on the z=0 plane)
    std::vector<cv::Point3f> objPts;
    for (int i = 0; i < boardSize.height; ++i) {
        for (int j = 0; j < boardSize.width; ++j) {
            objPts.push_back(cv::Point3f(j * squareSize, i * squareSize, 0));
        }
    }


    // Load image paths from chess_image_list.xml
    cv::FileStorage fs("../data/chess_image_list.xml", cv::FileStorage::READ);
    std::vector<std::string> imageFiles;
    fs["images"] >> imageFiles;

    if (imageFiles.empty()) {
        std::cerr << "Error: No images found in chess_image_list.xml" << std::endl;
        return -1;
    }

    // Iterate through each image
    for (const auto& imageFile : imageFiles) {
        cv::Mat img = cv::imread(imageFile);
        if (img.empty()) {
            std::cerr << "Error loading image: " << imageFile << std::endl;
            return -1;
        }
        
        cv::Mat grayImg;
        
        cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
        
        cv::GaussianBlur(grayImg, grayImg, cv::Size(5, 5), 0);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(grayImg, boardSize, corners, 
                                    cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);

        if (found) {
            // Refine corner locations
            cv::cornerSubPix(grayImg, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

	
            imagePoints.push_back(corners);
            objectPoints.push_back(objPts);

            // Draw the corners for visualization
            cv::drawChessboardCorners(img, boardSize, corners, found);
            //cv::imshow("Chessboard", img); //commented out
            //cv::waitKey(100); // Show each image for 100ms (adjust if needed) //commented out
        } else {
            std::cerr << "Chessboard not found in image: " << imageFile << std::endl;
        }
    }

    // Camera calibration
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F); // Initial camera matrix (identity matrix)
    cv::Mat distCoeffs = cv::Mat::zeros(8, 1, CV_64F); // No distortion initially
    std::vector<cv::Mat> rvecs, tvecs;
    
    // Calibrate the camera using the collected points
    double rms = cv::calibrateCamera(objectPoints, imagePoints, boardSize, cameraMatrix, distCoeffs, rvecs, tvecs);
    std::cout << "Calibration done with RMS error: " << rms << std::endl;

    // Save the camera parameters to a file
    cv::FileStorage fs_out("camera_params.xml", cv::FileStorage::WRITE);
    fs_out << "camera_matrix" << cameraMatrix;
    fs_out << "distortion_coefficients" << distCoeffs;
    fs_out.release();

    std::cout << "Camera calibration completed and saved to 'camera_params.xml'" << std::endl;
    return 0;
}

