-- Author: Robert Krook (guskrooro@student.gu.se)
-- You may use this code for educational purposes if you keep these three lines
-- as they are currently.

import "/util"

-- used in context with the gaussian blur further down, taken from
-- wikipedia https://en.wikipedia.org/wiki/Gaussian_blur
let gaussian_function (x: f32) (y: f32) (std_dev: f32): f32 =
  let op1 = f32.exp(-((x*x + y*y)/(2f32*(std_dev*std_dev))))
  let op2 = 1f32/(2f32*f32.pi*(std_dev*std_dev))
  in op1*op2

-- float version of 3by3 convolution on images. Pads borders with zeros to
-- not index out of bounds.
let convolve3by3 [n][m]
             (inp: [n][m]i32)
             (filt: [3][3]f32):
             [n][m]i32 =
  -- pad input with zeros to not index out of bounds
  let pad_inp = 
      [replicate (m+2) 0f32] ++ 
        (map (\row -> [0f32] ++ (map r32 row) ++ [0f32]) inp) ++ 
          [replicate (m+2) 0f32]

  -- this is prettier on the eyes, but runs good only on futhark-c
  --in map (\i -> map (\j -> loop res = 0 for j' in [-1,0,1] do
  --                             loop interim = 0 for i' in [-1,0,1] do
  --                                 interim + pad_inp[i+i',j+j']*
  --                                           filt[i'+1,j'+1]) 
  --             (tail (iota (m+1))))
  --   (tail (iota (n+1)))

  -- this runs well on futhark-opencl
  in map (\i -> map (\j -> i32.f32(
                           pad_inp[i-1,j-1]*filt[0,0] + 
                           pad_inp[i-1,j]*filt[0,1]   +
                           pad_inp[i-1,j+1]*filt[0,2] + 
                           pad_inp[i,j-1]*filt[1,0]   +
                           pad_inp[i,j]*filt[1,1]     +
                           pad_inp[i,j+1]*filt[1,2]   +
                           pad_inp[i+1,j-1]*filt[2,0] +
                           pad_inp[i+1,j]*filt[2,1]   +
                           pad_inp[i+1,j+1]*filt[2,2]))
               (tail (iota (m+1))))
     (tail (iota (n+1)))
     
-- integer version executes slightly quicker
let convolve3by3_i32 [n][m]
             (inp: [n][m]i32)
             (filt: [3][3]i32):
             [n][m]i32 =
  let pad_inp = 
      [replicate (m+2) 0i32] ++ 
        (map (\row -> [0i32] ++ row ++ [0i32]) inp) ++ 
          [replicate (m+2) 0i32]
  in map (\i -> map (\j -> pad_inp[i-1,j-1]*filt[0,0] + 
                           pad_inp[i-1,j]*filt[0,1]   +
                           pad_inp[i-1,j+1]*filt[0,2] + 
                           pad_inp[i,j-1]*filt[1,0]   +
                           pad_inp[i,j]*filt[1,1]     +
                           pad_inp[i,j+1]*filt[1,2]   +
                           pad_inp[i+1,j-1]*filt[2,0] +
                           pad_inp[i+1,j]*filt[2,1]   +
                           pad_inp[i+1,j+1]*filt[2,2])
               (tail (iota (m+1))))
     (tail (iota (n+1)))

-- follows the same pattern as convolve3by3.
let convolve5by5 [n][m]
             (inp: [n][m]i32)
             (filt: [5][5]f32):
             [n][m]i32 =
  let pad_inp = [replicate (m+4) 0f32, replicate (m+4) 0f32] ++ 
                (map (\row -> [0f32,0f32]       ++ 
                              (map r32 row)     ++ 
                              [0f32,0f32]) inp) ++ 
                [replicate (m+4) 0f32, replicate (m+4) 0f32]
  in map (\i -> map (\j -> i32.f32(
                  pad_inp[i-2,j-2]*filt[0,0] + pad_inp[i-2,j-1]*filt[0,1] +
                  pad_inp[i-2,j]*filt[0,2] + pad_inp[i-2,j+1]*filt[0,3] +
                  pad_inp[i-2,j+2]*filt[0,4] + 
                  pad_inp[i-1,j-2]*filt[1,0] + pad_inp[i-1,j-1]*filt[1,1] + 
                  pad_inp[i-1,j]*filt[1,2] + pad_inp[i-1,j+1]*filt[1,3] + 
                  pad_inp[i-1,j+2]*filt[1,4] +
                  pad_inp[i,j-2]*filt[2,0] + pad_inp[i,j-1]*filt[2,1] + 
                  pad_inp[i,j]*filt[2,2] + pad_inp[i,j+1]*filt[2,3] + 
                  pad_inp[i,j+2]*filt[2,4] +
                  pad_inp[i+1,j-2]*filt[3,0] + pad_inp[i+1,j-1]*filt[3,1] + 
                  pad_inp[i+1,j]*filt[3,2] + pad_inp[i+1,j+1]*filt[3,3] + 
                  pad_inp[i+1,j+2]*filt[3,4] +
                  pad_inp[i+2,j-2]*filt[4,0] + pad_inp[i+2,j-1]*filt[4,1] + 
                  pad_inp[i+2,j]*filt[4,2] + pad_inp[i+2,j+1]*filt[4,3] + 
                  pad_inp[i+2,j+2]*filt[4,4]
                ))
                (tail (tail (iota (m+2))))) 
     (tail (tail (iota (n+2))))
     
-- applies a gaussian blur to the input, smoothing out any edges. Goal of this
-- operation is to eliminate noises caused by false edges.
let gaussian_blur [n][m] (inp: [n][m]i32) (std_dev: f32) : [n][m]i32 =

  -- builds the 5by5 gaussian filter to convolve over the image
  let filt = map (\i -> map (\j -> gaussian_function (r32 (2-i)) (r32 (2-j))
                            std_dev)
                       (iota 5))
             (iota 5)
  in convolve5by5 inp filt

-- computes gradient intensity and gradient angle from the given image
let gradient_intensity [n][m] 
                       (inp: [n][m]i32):
                       [n][m](i32, i32) =
  let filt_x = [[-1i32, 0i32,  1i32],
                [-2i32, 0i32,  2i32],
                [-1i32, 0i32,  1i32]]
  let filt_y = [[-1i32,-2i32,-1i32],
                [ 0i32, 0i32, 0i32],
                [ 1i32, 2i32, 1i32]]
  let g_x = convolve3by3_i32 inp filt_x
  let g_y = convolve3by3_i32 inp filt_y
  in map2 (\row1 row2 -> 
               map2 (\pix1 pix2 -> 
                   -- calculate the sum of the squared gradients
                   let p1sqr = (r32 pix1) * (r32 pix1)
                   let p2sqr = (r32 pix2) * (r32 pix2)
                   let sqrsum = i32.f32 (f32.sqrt(p1sqr + p2sqr))

                   -- first aquire angle in radians in interval [-pi, pi]
                   let angle_rad = f32.atan2 (r32 pix1) (r32 pix2)

                   -- then turn angle to degrees in [-180,180], and then
                   -- then that interval into [0,360]
                   let angle_deg = angle_rad * (180f32 / f32.pi) + 180f32

                   -- finally round the degree to the closest value
                   -- in [0, 45, 90, 135]
                   let final_angle = -- 0 degrees 
                                    if (angle_deg >= 337.5f32 || 
                                         angle_deg < 22.5f32) ||
                                         (angle_deg >= 157.5f32 &&
                                         angle_deg < 202.5f32)
                                     then 0i32
                                     else
                                     -- 45 degrees
                                     -- since origin is at top left corner,
                                     -- this is actually 135 degrees
                                     if (angle_deg >= 22.5f32 && 
                                        angle_deg < 67.5f32) ||
                                        (angle_deg >= 202.5f32 &&
                                        angle_deg < 247.5f32)
                                     then 135i32
                                     else
                                     -- 90 degrees
                                     if (angle_deg >= 67.5f32 &&
                                        angle_deg < 112.5f32) ||
                                        (angle_deg >= 247.5f32 &&
                                        angle_deg < 292.5f32)
                                     then 90i32
                                     else
                                     -- 135 degrees
                                     -- since origin is at top left corner,
                                     -- this is actually 45 degrees
                                     if (angle_deg >= 112.5f32 &&
                                        angle_deg < 157.5f32) ||
                                        (angle_deg >= 292.5f32 &&
                                        angle_deg < 337.5f32)
                                     then 45i32
                                     else 0i32
                   -- given the way there is no branching on GPUs i imagine
                   -- this introduces a bit of inefficiency.
                   in (sqrsum, final_angle))
               row1 row2)
     g_x g_y

-- the supression operation used in non-maximum-supression further down.
-- The gradient direction is the direction orthogonal to the edge, so if the
-- angle is i.e 0 degrees, the edge is pointing north-south ward. We consider
-- THIS pixel a strong one if it is stronger than the pixels in both
-- directions orthogonal to the gradient direction.
let supression (inp: [3][3](i32, i32)): i32 =
  -- gradients to the northwest, north and norteast
  let ((nw_g,_), (n_g,_) ,(ne_g,_)) = (inp[0,0], inp[0,1], inp[0,2])

  -- gradients to the west, center and east
  let ((w_g,_),  (c_g,c_a), (e_g,_))  = (inp[1,0], inp[1,1], inp[1,2])

  -- gradients to the southwest, south and southeast
  let ((sw_g,_), (s_g,_) ,(se_g,_)) = (inp[2,0], inp[2,1], inp[2,2])

  in      if c_a == 0i32 then   if c_g >= w_g  && c_g >= e_g  
                                then c_g 
                                else 0i32
     else if c_a == 45i32 then  if c_g >= ne_g && c_g >= sw_g 
                                then c_g 
                                else 0i32
     else if c_a == 90i32 then  if c_g >= n_g  && c_g >= s_g  
                                then c_g 
                                else 0i32
     else if c_a == 135i32 then if c_g >= nw_g && c_g >= se_g 
                                then c_g 
                                else 0i32
     else c_g

-- applies non-maximum supression to the image
let non_maximum_supression [n][m]
                             (inp: [n][m](i32, i32)):
                             [n][m]i32 =
  -- pad the input so we don't index out of bounds
  let pad_inp = [replicate (m+2) (0,0)] ++ 
                (map (\row -> [(0,0)] ++ row ++ [(0,0)]) inp) ++ 
                [replicate (m+2) (0,0)]
  
  -- for each pixel, apply non-maximum supression by inspecting its neighbours
  in map (\i -> map (\j -> supression 
                       [[pad_inp[i-1,j-1], pad_inp[i-1,j], pad_inp[i-1,j+1]],
                        [pad_inp[i,j-1],   pad_inp[i,j],   pad_inp[i,j+1]],
                        [pad_inp[i+1,j-1], pad_inp[i+1,j], pad_inp[i+1,j+1]]])
                (tail (iota (m+1))))
     (tail (iota (n+1)))

-- applies double thresholding and 'edge tracking by hysteresis'
-- implemented after the wikipedia description of the algorithm.
let double_threshold [n][m]
                       (inp: [n][m]i32)
                       (upper: i32)
                       (lower: i32):
                       [n][m]i32 =
  -- pad the input so we don't index out of bounds
  let pad_inp = [replicate (m+2) 0] ++ 
                (map (\row -> [0] ++ row ++ [0]) inp) ++ 
                [replicate (m+2) 0]

                           -- if pixel is less than the lower bound,
                           -- supress it
  in map (\i -> map (\j -> if pad_inp[i,j] < lower
                           then 0i32

                           -- else if the pixel is higher than the upper bound
                           -- it is a strong edge and should be kept
                           else if pad_inp[i,j] > upper
                                then pad_inp[i,j]

                                -- otherwise, we have a weak edge. We only
                                -- want to keep weak edges if they are
                                -- immediately connected to a strong edge.
                                else if pad_inp[i-1,j-1] > upper ||
                                        pad_inp[i-1,j]   > upper ||
                                        pad_inp[i-1,j+1] > upper ||
                                        pad_inp[i,j-1]   > upper ||
                                        pad_inp[i,j+1]   > upper ||
                                        pad_inp[i+1,j-1] > upper ||
                                        pad_inp[i+1,j]   > upper ||
                                        pad_inp[i+1,j+1] > upper
                                        then pad_inp[i,j]
                                     else 0i32)
                (tail (iota (m+1)))) 
     (tail (iota (n+1)))

-- and finally, we have all the components of the canny edge
-- detection algorithm.
entry canny_edge_detection [n][m] 
                         (inp: [n][m]i32)
                         (std_dev: f32)
                         (upper: i32)
                         (lower: i32): [n][m]i32 =
  let blurred = gaussian_blur inp std_dev
  let grad_int = gradient_intensity blurred
  let supressed = non_maximum_supression grad_int
  in double_threshold supressed upper lower
