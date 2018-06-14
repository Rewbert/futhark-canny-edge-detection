-- Author: Robert Krook (guskrooro@student.gu.se)
-- You may use this code for educational purposes if you keep these three lines
-- as they are currently.

module Util = {

-- basic grayscaling. Equation picked up from wikipedia
-- https://en.wikipedia.org/wiki/Grayscale
entry grayscale [n][m] (inp: [n][m][3]i32): [n][m]i32 =
    map (\row ->
      map (\pixel -> i32.f32(0.299f32*(r32 pixel[0]) +
                             0.587f32*(r32 pixel[1]) +
                             0.114f32*(r32 pixel[2])))
      row)
    inp

-- got the following two functions from Troels Henriksen, one of the creators
-- of futhark, via slack communications. Many thanks as these by far
-- outperform what i tried.
let histogram_seq [n] (k: i32) (is: [n]i32): [k]i32 =
  loop h = replicate k 0 for i in is do
    if i >= 0 && i < k then unsafe h with [i] <- h[i] + 1
                       else h

let histogram [n][m] (k: i32) (is: [n][m]i32): [k]i32 =
  stream_red_per (map2 (+)) (histogram_seq k) (flatten is)

-- equalizes the histogram of an image. Effectively makes light pixels
-- lighter and dark ones darker.
entry histequalization [n][m] (inp: [n][m]i32) (num_vals: i32): [n][m]i32 =
  let cdf = scan (+) 0 (histogram num_vals inp)
  let cdf_min = reduce (\x y -> if x != 0 then x else y) 0 cdf
  let h = (\v -> i32.f32 ((r32 (cdf[v] - cdf_min) / 
                           r32 (n*m - cdf_min)) * 
                           r32 (num_vals - 1)))
  in map (\row -> map (\pixel -> unsafe h pixel) row) inp

-- simple thresholding, binarizes an image after the given threshold
entry binarization [n][m] (inp: [n][m]i32) (threshold: i32): [n][m]i32 =
  map (\row -> 
     map (\pixel -> if pixel >= threshold then 255i32 else 0i32)
     row)
  inp
}
