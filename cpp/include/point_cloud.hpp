

#ifndef POINT_CLOUD_HPP
#define POINT_CLOUD_HPP

#include <cstdint>
namespace point_cloud_data {

/**
 * A point cloud tensor represents a point cloud and has the following values:
 *
 * x: The x coordinate
 * y: The y coordinate
 * z: The z coordinate
 * i: The intencity
 */

#define POINT_CLOUD_X_IDX 0
#define POINT_CLOUD_Y_IDX 1
#define POINT_CLOUD_Z_IDX 2
#define POINT_CLOUD_I_IDX 3

/**
 * Defines options for maximum intensity values.
 * Intensity goes from [0; MAX_INTENSITY], where MAX_INTENSITY is either 1 or
 * 255.
 */
enum struct intensity_range  : std::uint8_t {
  MAX_INTENSITY_1 = 1,
  MAX_INTENSITY_255 = 255,
};

namespace max_intensity {

static uint8_t intensity =
    static_cast<uint8_t>(intensity_range::MAX_INTENSITY_1);

inline void set(intensity_range val) { intensity = static_cast<uint8_t>(val); }
[[nodiscard]] inline uint8_t get() { return intensity; }

} // namespace max_intensity

} // namespace point_cloud_data

#endif // !POINT_CLOUD_HPP
