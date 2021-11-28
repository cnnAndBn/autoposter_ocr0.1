// This implementation is from https://github.com/whai362/PSENet/blob/master/pse/adaptor.cpp

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include <iostream>
#include <queue>

using namespace std;

namespace py = pybind11;

namespace pse_adaptor {

    class Point2d {
    public:
        int x;
        int y;

        Point2d() : x(0), y(0)
        {}

        Point2d(int _x, int _y) : x(_x), y(_y)
        {}
    };

    void growing_text_line(const int *data,
                           vector<long int> &data_shape,
                           const int *label_map,
                           vector<long int> &label_shape,
                           int &label_num,    //label_num:cca的聚类结果数，包括背景0这类
                           float &min_area,
                           vector<vector<int>> &text_line) {
        int area[label_num + 1];
        memset(area, 0, sizeof(area));
        //统计cca聚类结果，每类的面积大小
        for (int x = 0; x < label_shape[0]; ++x) {
            for (int y = 0; y < label_shape[1]; ++y) {
                int label = label_map[x * label_shape[1] + y];
                if (label == 0) continue;
                area[label] += 1;
            }
        }

        queue<Point2d> queue, next_queue;  //对cca的结果处理，即kernel最小的那个图
        for (int x = 0; x < label_shape[0]; ++x) {   //每行
            vector<int> row(label_shape[1]);
            for (int y = 0; y < label_shape[1]; ++y) {   //每列
                int label = label_map[x * label_shape[1] + y];   //cca结果取出
                if (label == 0) continue;
                if (area[label] < min_area) continue;

                Point2d point(x, y);
                queue.push(point);   //按从左到右，从上到小的顺序把满足面积大小要求的所有点的坐标存入queue中
                row[y] = label;      //满足面积大小要求的每行点的label信息
            }
            text_line.emplace_back(row);
        }

        int dx[] = {-1, 1, 0, 0};  //四领域xy坐标，x这里是行，y代表列，因此分别是上，下，左，右
        int dy[] = {0, 0, -1, 1};

        for (int kernel_id = data_shape[0] - 2; kernel_id >= 0; --kernel_id) {   //kernel map的循环，-2是因为label_map就是cca对最小那个kernel，也就是在第 data_shape[0]-1 那个图上做的，因此从-2开始
            while (!queue.empty()) {
                Point2d point = queue.front();
                queue.pop();
                int x = point.x;
                int y = point.y;
                int label = text_line[x][y];   //cca聚类的label

                bool is_edge = true;
                for (int d = 0; d < 4; ++d) {  //对该点的四邻域进行遍历
                    int tmp_x = x + dx[d];
                    int tmp_y = y + dy[d];

                    if (tmp_x < 0 || tmp_x >= (int)text_line.size()) continue;  //判断领域点是否超出图像边界
                    if (tmp_y < 0 || tmp_y >= (int)text_line[1].size()) continue;//判断领域点是否超出图像边界
                    int kernel_value = data[kernel_id * data_shape[1] * data_shape[2] + tmp_x * data_shape[2] + tmp_y];  //取出预测的kernel在该邻域坐标的结果
                    if (kernel_value == 0) continue;  //网络预测为背景的情况
                    if (text_line[tmp_x][tmp_y] > 0) continue;  //说明该领域点在更小的kernel图上已经被指派给某个聚类了，否则为网络在该kernel下预测为前景，且没有被更小的kernel预测为前景的点
                                                                //要将这样的加入，label为该邻域中心点的label

                    Point2d point(tmp_x, tmp_y);
                    queue.push(point);
                    text_line[tmp_x][tmp_y] = label;
                    is_edge = false;
                }

                if (is_edge) {
                    next_queue.push(point);  //为该kernel下的边缘点（即存在四邻域有领域点为0的情况，即在该kernel尺寸下其存在邻域点为背景0的情况，即第81行的case），则将其加入到next_queue中用于在下一个更大的kernel上进行expansion扩充操作
                }
            }
            swap(queue, next_queue);
        }
    }


    vector<vector<int>> pse(py::array_t<int, py::array::c_style | py::array::forcecast> quad_n9,
                            float min_area,
                            py::array_t<int32_t, py::array::c_style> label_map,
                            int label_num) {
        auto buf = quad_n9.request();
        auto data = static_cast<int *>(buf.ptr);
        vector<long int> data_shape = buf.shape;

        auto buf_label_map = label_map.request();
        auto data_label_map = static_cast<int32_t *>(buf_label_map.ptr);
        vector<long int> label_map_shape = buf_label_map.shape;

        vector<vector<int>> text_line;

        growing_text_line(data,
                          data_shape,
                          data_label_map,
                          label_map_shape,
                          label_num,
                          min_area,
                          text_line);

        return text_line;
    }
}

PYBIND11_PLUGIN(pse) {
    py::module m("pse", "pse");

    m.def("pse", &pse_adaptor::pse, "pse");

    return m.ptr();
}
