import cv2
import numpy as np
import matplotlib.pyplot as plt
from lane_tracking.util.carla_util import carla_img_to_array


def threshold_binary(img):
    """
    param: img - raw image
    return: theshold_binary
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # red channel - side
    r_channel = img[:,:,2]
    r_thresh_min = 200
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel >= r_thresh_min)] = 1

    # green channel - road
    g_channel = img[:,:,1]
    g_thresh_min = 230
    g_thresh_max = 235
    g_binary = np.zeros_like(g_channel)
    g_binary[(g_channel >= g_thresh_min) & (g_binary <= g_thresh_max)] = 1

    # combined
    combined_binary = np.zeros_like(r_channel)
    combined_binary[(r_binary == 1) | (g_binary == 1)] = 1
    combined_binary = cv2.Canny(combined_binary, 0.5, 1)
    return combined_binary


def four_point_transform(image):
    """
    points order: tl, tr, br, bl
    """
    # rect = np.array([[337, 144], [419, 144], [668, 257], [100, 257]], dtype="float32")  # sharp curve right
    rect = np.array([[293, 166], [440, 166], [567, 257], [100, 257]], dtype="float32")
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    M_inv = cv2.getPerspectiveTransform(dst, rect)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped, M_inv


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 15
    # Set the width of the windows +/- margin
    margin = 30
    # Set minimum number of pixels found to recenter window
    minpix = 5
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped, leftx, lefty, rightx, righty):
    # if no lane detected, use other lane data to project
    if not leftx.any() or not lefty.any():
        lefty = righty
        leftx = rightx - 275
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    center_fit = (left_fit+right_fit)*0.5
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        center_fitx = center_fit[0] * ploty ** 2 + center_fit[1] * ploty + center_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty
        center_fitx = 1 * ploty ** 2 + 1 * ploty

    # bad result: left & right cross-over or left & right too wide
    if max(left_fitx) > max(right_fitx):
        left_fitx = right_fitx - 275
        center_fitx = right_fitx - 135
    elif np.absolute(np.mean(left_fitx) - np.mean(right_fitx)) > 325:
        left_fitx = right_fitx - 275
        center_fitx = right_fitx - 135
    elif np.absolute(np.mean(left_fitx) - np.mean(right_fitx)) < 250:
        left_fitx = right_fitx - 275
        center_fitx = right_fitx - 135

    return ploty, left_fit, right_fit, left_fitx, right_fitx, center_fitx


def measure_position_meters(binary_warped, left_fit, right_fit):
    # Define conversion in x from pixels space to meters
    xm_per_pix = 3.7/275 # meters per pixel in x dimension
    # Choose the y value corresponding to the bottom of the image
    y_max = binary_warped.shape[0]
    # Calculate left and right line positions at the bottom of the image
    left_x_pos = left_fit[0]*y_max**2 + left_fit[1]*y_max + left_fit[2]
    right_x_pos = right_fit[0]*y_max**2 + right_fit[1]*y_max + right_fit[2]
    # Calculate the x position of the center of the lane
    center_lanes_x_pos = (left_x_pos + right_x_pos)//2
    # Calculate the deviation between the center of the lane and the center of the picture
    # The car is assumed to be placed in the center of the picture
    # If the deviation is negative, the car is on the felt hand side of the center of the lane
    veh_pos = ((binary_warped.shape[1]//2) - center_lanes_x_pos) * xm_per_pix
    return veh_pos


def measure_curvature_meters(left_fitx, right_fitx, ploty):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 250  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 275  # meters per pixel in x dimension

    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    return left_curverad, right_curverad


def warp_back(img, bird_eye, ploty, left_fitx, right_fitx, M_inv, left_curverad, right_curverad, veh_pos):
    warp_zero = np.zeros_like(bird_eye).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, M_inv, (img.shape[1], img.shape[0]))
    out_img = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    cv2.putText(out_img, 'Radius (m): ' + str((left_curverad + right_curverad) / 2)[:5], (10, 20),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    cv2.putText(out_img, 'Off-Center (m): ' + str(veh_pos)[:5], (10, 40), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 255, 255), 2)
    return out_img


def image_pipeline(image):
    img = carla_img_to_array(image)
    # img=image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # undistort + binary threshold
    binary_img = threshold_binary(img)
    # bird eye transformation
    bird_eye, M_inv = four_point_transform(binary_img)
    # find lane pixels
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(bird_eye)
    # fit polynomial
    ploty, left_fit, right_fit, left_fitx, right_fitx = fit_polynomial(bird_eye, leftx, lefty, rightx, righty)
    # measure curvature + vehicle position
    left_curverad, right_curverad =  measure_curvature_meters(left_fitx, right_fitx, ploty)
    veh_pos = measure_position_meters(bird_eye, left_fit, right_fit)
    # warp back to original img
    new_img = warp_back(img, bird_eye, ploty, left_fitx, right_fitx, M_inv, left_curverad, right_curverad, veh_pos)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    return new_img


# img = cv2.imread('./data/dgmd/image_10.512749698013067.png')
# # img = cv2.imread('./data/dgmd/image_58.81275221705437.png')
# # img = cv2.imread('./data/dgmd/image_59.77941893413663.png')
# # img = cv2.imread('./data/dgmd/image_60.74608565121889.png')
# # img = cv2.imread('./data/dgmd/image_65.5794192366302.png')
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#
# binary_img = threshold_binary(img)
# bird_eye, M_inv = four_point_transform(binary_img)
# leftx, lefty, rightx, righty, out_img = find_lane_pixels(bird_eye)
# ploty, left_fit, right_fit, left_fitx, right_fitx, center_fitx = fit_polynomial(bird_eye, leftx, lefty, rightx, righty)
# left_curverad, right_curverad =  measure_curvature_meters(left_fitx, right_fitx, ploty)
# veh_pos = measure_position_meters(bird_eye, left_fit, right_fit)
# new_img = warp_back(img, bird_eye, ploty, left_fitx, right_fitx, M_inv, left_curverad, right_curverad, veh_pos)
#
# # x = np.arange(-2,60,1.0)
# # y = -0.5*(poly_left(x)+poly_right(x))
# # test = (left_fit+right_fit)*0.5
# # print(center_fitx)
#
# # # new_img = image_pipeline(img)
# plt.plot(left_fitx, ploty, color='yellow')
# plt.plot(right_fitx, ploty, color='yellow')
# plt.imshow(new_img)
# plt.show()
