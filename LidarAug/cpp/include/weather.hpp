
#ifndef WEATHER_HPP
#define WEATHER_HPP

#include <cstddef>
#include <torch/serialize/tensor.h>

typedef enum { DIST, CHAMFER } fog_metric;

void fog(at::Tensor point_cloud, std::size_t prob, fog_metric metric,
         float sigma, int mean);

#endif // !WEATHER_HPP
