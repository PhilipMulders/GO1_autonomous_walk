#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

// Function to generate 3D points of the chessboard
std::vector<cv::Point3f> generateBoardPoints(cv::Size boardSize, float squareSize) {
    std::vector<cv::Point3f> boardPoints;
    for (int i = 0; i < boardSize.height; ++i) {
        for (int j = 0; j < boardSize.width; ++j) {
            boardPoints.push_back(cv::Point3f(j * squareSize, i * squareSize, 0));
        }
    }
    return boardPoints;
}

// Function to compute reprojection error
double computeReprojectionErrors(
    const std::vector<std::vector<cv::Point3f>>& objectPoints,
    const std::vector<std::vector<cv::Point2f>>& imagePoints,
    const std::vector<cv::Mat>& rvecs,
    const std::vector<cv::Mat>& tvecs,
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs) {

    double totalError = 0;
    int totalPoints = 0;
    for (size_t i = 0; i < objectPoints.size(); ++i) {
        std::vector<cv::Point2f> projectedPoints;
        cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, projectedPoints);
        double error = cv::norm(imagePoints[i], projectedPoints, cv::NORM_L2);
        totalError += error * error;
        totalPoints += objectPoints[i].size();
    }
    return std::sqrt(totalError / totalPoints);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./pose_estimation <image_path>" << std::endl;
        return -1;
    }

    // Chessboard parameters
    cv::Size boardSize(9, 6);  // 9x6 board, adjust as per your chessboard
    float squareSize = 3.0; // 3 cm

    // Load the camera calibration parameters
    cv::FileStorage fs("camera_params.yaml", cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open camera parameters file!" << std::endl;
        return -1;
    }
    cv::Mat cameraMatrix, distCoeffs;
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();

    // Load the test image for pose estimation
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    // Detect chessboard corners
    std::vector<cv::Point2f> ptvec;
    bool found = cv::findChessboardCorners(img, boardSize, ptvec, cv::CALIB_CB_ADAPTIVE_THRESH);
    if (!found) {
        std::cerr << "Chessboard not found!" << std::endl;
        return -1;
    }

    // Refine the corner locations for better accuracy
    cv::cornerSubPix(img, ptvec, cv::Size(11, 11), cv::Size(-1, -1), 
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

    // Generate 3D points for the chessboard
    std::vector<cv::Point3f> boardPoints = generateBoardPoints(boardSize, squareSize);

    // Solve PnP to estimate the pose
    cv::Mat rvec, tvec;
    cv::solvePnP(boardPoints, ptvec, cameraMatrix, distCoeffs, rvec, tvec);

    // Calculate the distance from the camera to the chessboard
    double distance = cv::norm(tvec);
    std::cout << "Distance to chessboard: " << distance << " units" << std::endl;

    // Reprojection error (optional, for debugging the accuracy)
    std::vector<std::vector<cv::Point3f>> objectPoints(1, boardPoints);
    std::vector<std::vector<cv::Point2f>> imagePoints(1, ptvec);
    std::vector<cv::Mat> rvecs(1, rvec), tvecs(1, tvec);
    double reprojectionError = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix, distCoeffs);
    std::cout << "Reprojection error: " << reprojectionError << std::endl;

    return 0;
}
