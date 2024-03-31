
#ifndef LABEL_HPP
#define LABEL_HPP

/**
 * A label tensor represents a box and has the following values:
 *
 * x:     The x coordinate (center of the box)
 * y:     The y coordinate (center of the box)
 * z:     The z coordinate (center of the box)
 * w:     The width  of the box (along the x axis)
 * h:     The height of the box (along the z axis)
 * l:     The length of the box (along the y axis)
 * Theta: The rotation angle
 */

#define LABEL_X_IDX 0
#define LABEL_Y_IDX 1
#define LABEL_Z_IDX 2
#define LABEL_W_IDX 3
#define LABEL_H_IDX 4
#define LABEL_L_IDX 5
#define LABEL_ANGLE_IDX 6
#define LABEL_NUM_FEATURES 7

#endif // !LABEL_HPP
