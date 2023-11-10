#include <pybind11/pybind11.h>
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <iostream>

typedef struct{float x,y,z;} vec;
typedef struct{int batch_size, num_points, num_point_features;} dimensions;
typedef struct{float scale; vec translate, rotate;} transformations;


void translate(at::Tensor points, at::Tensor translation){
    dimensions dims = {points.size(0), points.size(1), points.size(2)};
    float *t = translation.data<float>();
    vec translate = {t[0], t[1], t[2]};

    //translate all point clouds in a batch by the same amount
    for(int i = 0; i < dims.batch_size; i++){
       for(int j = 0; j < dims.num_points; j++){
            points.index({i, j, 0}) += translate.x;
            points.index({i, j, 1}) += translate.y;
            points.index({i, j, 2}) += translate.z;
        } 
    }

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("translate", &translate, "translation function for point clouds in C++");
}

