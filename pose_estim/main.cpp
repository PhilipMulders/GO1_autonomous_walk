#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // Load camera calibration parameters (intrinsics and distortion)
    cv::Mat cameraMatrix, distCoeffs;
    cv::FileStorage fs("../camera_params.xml", cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error: Could not open the camera calibration file!" << std::endl;
        return -1;
    }
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;

    // Load the input image (grayscale)
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Chessboard size (number of inner corners per row and column)
    cv::Size boardSize(7, 7);
    std::vector<cv::Point2f> corners;
    
    // Detect the chessboard corners in the image
    bool found = cv::findChessboardCorners(img, boardSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    if (!found) {
        std::cerr << "Chessboard not found in the image!" << std::endl;
        return -1;
    }

    // Refine corner positions to subpixel accuracy
    cv::cornerSubPix(img, corners, cv::Size(11, 11), cv::Size(-1, -1),
                 cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));

    // Generate 3D points for the chessboard corners (in world coordinates)
    std::vector<cv::Point3f> objectPts;
    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {
            objectPts.push_back(cv::Point3f(j * 2.0f, i * 2.0f, 0.0f)); // 2 cm square size
        }
    }

    // SolvePnP to get the rotation and translation vectors
    cv::Mat rvec, tvec;
    solvePnP(objectPts, corners, cameraMatrix, distCoeffs, rvec, tvec);

    // Project the 3D points back to 2D for visualization and error computation
    std::vector<cv::Point2f> projectedPoints;
    projectPoints(objectPts, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

    // Compute the reprojection error
    double error = 0.0;
    for (size_t i = 0; i < corners.size(); i++) {
        error += norm(corners[i] - projectedPoints[i]);
    }
    std::cout << "Reprojection error: " << error / corners.size() << std::endl;

    // Calculate the distance to a specific chessboard corner (e.g., top-left corner)
    cv::Point3f corner3D = objectPts[0];  // Top-left corner
    cv::Point3f cameraPosition(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

    // Calculate Euclidean distance from the camera to the corner
    double distance = norm(cameraPosition - corner3D);
    std::cout << "Distance to corner: " << distance << " cm" << std::endl;

    // Draw the detected corners and display the image
    cv::Mat imgWithCorners;
    cv::drawChessboardCorners(img, boardSize, corners, found);
    //imshow("Chessboard Detection", imgWithCorners);
    //waitKey(0);

    return 0;
}
