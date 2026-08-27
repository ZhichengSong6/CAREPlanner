#include <ros/ros.h>

#include <care_collision_cdf/CollisionCDFConstraintBatch.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/String.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace care_collision_cdf {
namespace {

using Clock = std::chrono::steady_clock;

double msBetween(const Clock::time_point& a, const Clock::time_point& b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

bool finiteFloat(float v) {
  return std::isfinite(static_cast<double>(v));
}

struct Anchor {
  std::array<float, 3> center{};
  std::array<float, 7> q{};
  float radius = 0.0f;
  int eval_timestep = -1;
  int original_timestep = -1;
};

struct MapIndex {
  std::vector<uint8_t> low;
  std::vector<float> confidence;
  std::vector<float> current_visibility;
  ros::Time stamp;
  ros::Time received;
  std::size_t total_voxel_count = 0;
  std::size_t low_voxel_count = 0;
  double build_ms = 0.0;
};

struct PairMeta {
  std::array<float, 3> point{};
  std::array<float, 7> q{};
  float confidence = 0.0f;
  float current_visibility = 0.0f;
  float approx_body_clearance_m = 0.0f;
  int eval_timestep = -1;
  int original_timestep = -1;
};

#pragma pack(push, 1)
struct RequestHeader {
  char magic[4];
  uint32_t pair_count;
};

struct ResponseHeader {
  char magic[4];
  uint32_t pair_count;
  double h2d_ms;
  double inference_ms;
  double d2h_ms;
  double worker_total_ms;
};
#pragma pack(pop)

static_assert(sizeof(RequestHeader) == 8, "Unexpected request-header packing");
static_assert(sizeof(ResponseHeader) == 40, "Unexpected response-header packing");

bool sendAll(int fd, const void* data, std::size_t bytes) {
  const auto* ptr = static_cast<const uint8_t*>(data);
  std::size_t sent = 0;
  while (sent < bytes) {
    const ssize_t n = ::send(fd, ptr + sent, bytes - sent, MSG_NOSIGNAL);
    if (n <= 0) {
      return false;
    }
    sent += static_cast<std::size_t>(n);
  }
  return true;
}

bool recvAll(int fd, void* data, std::size_t bytes) {
  auto* ptr = static_cast<uint8_t*>(data);
  std::size_t received = 0;
  while (received < bytes) {
    const ssize_t n = ::recv(fd, ptr + received, bytes - received, 0);
    if (n <= 0) {
      return false;
    }
    received += static_cast<std::size_t>(n);
  }
  return true;
}

struct Stats {
  double min = std::numeric_limits<double>::quiet_NaN();
  double mean = std::numeric_limits<double>::quiet_NaN();
  double max = std::numeric_limits<double>::quiet_NaN();
};

Stats simpleStats(const std::vector<float>& values) {
  Stats out;
  if (values.empty()) {
    return out;
  }
  double sum = 0.0;
  out.min = std::numeric_limits<double>::infinity();
  out.max = -std::numeric_limits<double>::infinity();
  for (float v : values) {
    const double x = static_cast<double>(v);
    out.min = std::min(out.min, x);
    out.max = std::max(out.max, x);
    sum += x;
  }
  out.mean = sum / static_cast<double>(values.size());
  return out;
}

}  // namespace

class CppForbiddenVoxelGpuShadow {
 public:
  CppForbiddenVoxelGpuShadow()
      : nh_(), pnh_("~") {
    pnh_.param<std::string>(
        "anchor_topic",
        anchor_topic_,
        "/care_planner/trajectory_risk/body_sweep_anchors");
    pnh_.param<std::string>(
        "map_topic",
        map_topic_,
        "/care_planner/confidence_map/points");
    pnh_.param<std::string>(
        "summary_topic",
        summary_topic_,
        "/care_planner/collision_cdf/cpp_gpu_online_summary");
    pnh_.param<std::string>(
        "constraint_batch_topic",
        constraint_batch_topic_,
        "/care_planner/collision_cdf/constraint_batch");
    pnh_.param<std::string>(
        "output_jsonl",
        output_jsonl_,
        "/tmp/c5_2h_cpp_gpu_online.jsonl");
    pnh_.param<std::string>(
        "gpu_socket",
        gpu_socket_,
        "/tmp/care_collision_cdf_gpu.sock");

    pnh_.param("rate", rate_hz_, 20.0);
    pnh_.param("anchor_stale_s", anchor_stale_s_, 0.25);
    pnh_.param("map_stale_s", map_stale_s_, 0.50);
    pnh_.param("confidence_threshold", confidence_threshold_, 0.50);
    pnh_.param("proximity_margin", proximity_margin_, 0.075);
    pnh_.param("max_pairs_per_step", max_pairs_per_step_, 250);
    pnh_.param("max_pairs", max_pairs_, 8000);
    pnh_.param("signed_zero_band", zero_band_, 0.05);

    pnh_.param("x_min", x_min_, -0.95);
    pnh_.param("x_max", x_max_, 0.95);
    pnh_.param("y_min", y_min_, -0.95);
    pnh_.param("y_max", y_max_, 0.95);
    pnh_.param("z_min", z_min_, 0.0);
    pnh_.param("z_max", z_max_, 1.15);
    pnh_.param("map_resolution", resolution_, 0.05);

    if (rate_hz_ <= 0.0 || resolution_ <= 0.0) {
      throw std::runtime_error("rate and resolution must be positive");
    }
    if (confidence_threshold_ < 0.0 || confidence_threshold_ > 1.0) {
      throw std::runtime_error("confidence threshold must be in [0,1]");
    }
    if (max_pairs_per_step_ <= 0 || max_pairs_ <= 0) {
      throw std::runtime_error("pair limits must be positive");
    }

    nx_ = static_cast<int>(
              std::floor((x_max_ - x_min_) / resolution_)) +
          1;
    ny_ = static_cast<int>(
              std::floor((y_max_ - y_min_) / resolution_)) +
          1;
    nz_ = static_cast<int>(
              std::floor((z_max_ - z_min_) / resolution_)) +
          1;
    if (nx_ <= 0 || ny_ <= 0 || nz_ <= 0) {
      throw std::runtime_error("invalid dense-grid dimensions");
    }
    grid_size_ = static_cast<std::size_t>(nx_) *
                 static_cast<std::size_t>(ny_) *
                 static_cast<std::size_t>(nz_);

    best_clearance_.assign(
        grid_size_, std::numeric_limits<float>::infinity());

    {
      std::ofstream out(output_jsonl_, std::ios::out | std::ios::trunc);
      if (!out) {
        throw std::runtime_error(
            "failed to open output_jsonl: " + output_jsonl_);
      }
    }

    summary_pub_ = nh_.advertise<std_msgs::String>(
        summary_topic_, 10);
    constraint_batch_pub_ =
        nh_.advertise<care_collision_cdf::CollisionCDFConstraintBatch>(
            constraint_batch_topic_, 2);
    anchor_sub_ = nh_.subscribe(
        anchor_topic_, 1,
        &CppForbiddenVoxelGpuShadow::anchorCallback, this);
    map_sub_ = nh_.subscribe(
        map_topic_, 1,
        &CppForbiddenVoxelGpuShadow::mapCallback, this);
    timer_ = nh_.createTimer(
        ros::Duration(1.0 / rate_hz_),
        &CppForbiddenVoxelGpuShadow::timerCallback, this);

    ROS_WARN_STREAM(
        "[C5.2h C++] READY rate=" << rate_hz_
        << " Hz grid=" << nx_ << "x" << ny_ << "x" << nz_
        << " (" << grid_size_ << " voxels)"
        << " conf<" << confidence_threshold_
        << " proximity_margin=" << proximity_margin_
        << " max_pairs_per_step=" << max_pairs_per_step_
        << " socket=" << gpu_socket_);
  }

  ~CppForbiddenVoxelGpuShadow() {
    closeGpuSocket();
  }

 private:
  int linearIndex(int ix, int iy, int iz) const {
    return ix * ny_ * nz_ + iy * nz_ + iz;
  }

  bool validIndex(int ix, int iy, int iz) const {
    return ix >= 0 && ix < nx_ &&
           iy >= 0 && iy < ny_ &&
           iz >= 0 && iz < nz_;
  }

  int coordToIndex(double x, double min_value, int n) const {
    const int i = static_cast<int>(
        std::llround((x - min_value) / resolution_));
    if (i < 0 || i >= n) {
      return -1;
    }
    return i;
  }

  std::array<float, 3> pointForIndex(int idx) const {
    const int ix = idx / (ny_ * nz_);
    const int rem = idx - ix * ny_ * nz_;
    const int iy = rem / nz_;
    const int iz = rem - iy * nz_;
    return {
        static_cast<float>(x_min_ + ix * resolution_),
        static_cast<float>(y_min_ + iy * resolution_),
        static_cast<float>(z_min_ + iz * resolution_)};
  }

  void anchorCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_anchor_cloud_ = msg;
    latest_anchor_received_ = ros::Time::now();
  }

  void mapCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    const auto t0 = Clock::now();

    auto index = std::make_shared<MapIndex>();
    index->low.assign(grid_size_, 0);
    index->confidence.assign(grid_size_, 0.0f);
    index->current_visibility.assign(grid_size_, 0.0f);
    index->stamp = msg->header.stamp;
    index->received = ros::Time::now();

    try {
      sensor_msgs::PointCloud2ConstIterator<float> ix_it(*msg, "x");
      sensor_msgs::PointCloud2ConstIterator<float> iy_it(*msg, "y");
      sensor_msgs::PointCloud2ConstIterator<float> iz_it(*msg, "z");
      sensor_msgs::PointCloud2ConstIterator<float> conf_it(*msg, "confidence");
      sensor_msgs::PointCloud2ConstIterator<float> vis_it(
          *msg, "current_visibility");

      for (; ix_it != ix_it.end();
           ++ix_it, ++iy_it, ++iz_it, ++conf_it, ++vis_it) {
        const float x = *ix_it;
        const float y = *iy_it;
        const float z = *iz_it;
        const float conf = *conf_it;
        const float vis = *vis_it;
        ++index->total_voxel_count;

        if (!finiteFloat(x) || !finiteFloat(y) || !finiteFloat(z) ||
            !finiteFloat(conf) || !finiteFloat(vis)) {
          continue;
        }

        const int ix = coordToIndex(x, x_min_, nx_);
        const int iy = coordToIndex(y, y_min_, ny_);
        const int iz = coordToIndex(z, z_min_, nz_);
        if (!validIndex(ix, iy, iz)) {
          continue;
        }

        const int linear = linearIndex(ix, iy, iz);
        index->confidence[static_cast<std::size_t>(linear)] = conf;
        index->current_visibility[static_cast<std::size_t>(linear)] = vis;
        if (static_cast<double>(conf) < confidence_threshold_) {
          index->low[static_cast<std::size_t>(linear)] = 1;
          ++index->low_voxel_count;
        }
      }
    } catch (const std::runtime_error& e) {
      ROS_ERROR_STREAM_THROTTLE(
          1.0, "[C5.2h C++] map PointCloud2 decode failed: " << e.what());
      return;
    }

    index->build_ms = msBetween(t0, Clock::now());

    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_map_ = index;
    }
  }

  std::vector<Anchor> decodeAnchors(
      const sensor_msgs::PointCloud2& cloud) const {
    std::vector<Anchor> anchors;
    anchors.reserve(cloud.width);

    try {
      sensor_msgs::PointCloud2ConstIterator<float> x_it(cloud, "x");
      sensor_msgs::PointCloud2ConstIterator<float> y_it(cloud, "y");
      sensor_msgs::PointCloud2ConstIterator<float> z_it(cloud, "z");
      sensor_msgs::PointCloud2ConstIterator<float> q0_it(cloud, "q0");
      sensor_msgs::PointCloud2ConstIterator<float> q1_it(cloud, "q1");
      sensor_msgs::PointCloud2ConstIterator<float> q2_it(cloud, "q2");
      sensor_msgs::PointCloud2ConstIterator<float> q3_it(cloud, "q3");
      sensor_msgs::PointCloud2ConstIterator<float> q4_it(cloud, "q4");
      sensor_msgs::PointCloud2ConstIterator<float> q5_it(cloud, "q5");
      sensor_msgs::PointCloud2ConstIterator<float> q6_it(cloud, "q6");
      sensor_msgs::PointCloud2ConstIterator<float> radius_it(
          cloud, "radius");
      sensor_msgs::PointCloud2ConstIterator<int32_t> eval_it(
          cloud, "eval_timestep");
      sensor_msgs::PointCloud2ConstIterator<int32_t> original_it(
          cloud, "original_timestep");

      for (; x_it != x_it.end();
           ++x_it, ++y_it, ++z_it,
           ++q0_it, ++q1_it, ++q2_it, ++q3_it,
           ++q4_it, ++q5_it, ++q6_it,
           ++radius_it, ++eval_it, ++original_it) {
        Anchor a;
        a.center = {*x_it, *y_it, *z_it};
        a.q = {
            *q0_it, *q1_it, *q2_it, *q3_it,
            *q4_it, *q5_it, *q6_it};
        a.radius = *radius_it;
        a.eval_timestep = *eval_it;
        a.original_timestep = *original_it;

        bool finite = finiteFloat(a.radius);
        for (float v : a.center) {
          finite = finite && finiteFloat(v);
        }
        for (float v : a.q) {
          finite = finite && finiteFloat(v);
        }
        if (!finite || a.radius <= 0.0f ||
            a.original_timestep < 0) {
          continue;
        }
        anchors.push_back(a);
      }
    } catch (const std::runtime_error& e) {
      ROS_ERROR_STREAM_THROTTLE(
          1.0, "[C5.2h C++] anchor PointCloud2 decode failed: " << e.what());
      anchors.clear();
    }
    return anchors;
  }

  std::vector<PairMeta> buildPairs(
      const std::vector<Anchor>& anchors,
      const MapIndex& map,
      std::size_t* raw_pair_count,
      int* active_step_count) {
    std::unordered_map<int, std::vector<const Anchor*>> by_step;
    by_step.reserve(32);
    for (const auto& anchor : anchors) {
      by_step[anchor.original_timestep].push_back(&anchor);
    }

    std::vector<int> steps;
    steps.reserve(by_step.size());
    for (const auto& item : by_step) {
      steps.push_back(item.first);
    }
    std::sort(steps.begin(), steps.end());

    std::vector<PairMeta> pairs;
    pairs.reserve(
        static_cast<std::size_t>(steps.size()) *
        static_cast<std::size_t>(max_pairs_per_step_));

    std::size_t raw_total = 0;
    int active_steps = 0;
    std::vector<int> touched;
    touched.reserve(1024);

    for (int step : steps) {
      const auto it = by_step.find(step);
      if (it == by_step.end() || it->second.empty()) {
        continue;
      }

      touched.clear();
      const Anchor& representative = *it->second.front();

      for (const Anchor* anchor_ptr : it->second) {
        const Anchor& anchor = *anchor_ptr;
        const double search_radius =
            static_cast<double>(anchor.radius) + proximity_margin_;
        const int n = static_cast<int>(
            std::ceil(search_radius / resolution_));

        const int cx = coordToIndex(anchor.center[0], x_min_, nx_);
        const int cy = coordToIndex(anchor.center[1], y_min_, ny_);
        const int cz = coordToIndex(anchor.center[2], z_min_, nz_);
        if (!validIndex(cx, cy, cz)) {
          continue;
        }

        const int x0 = std::max(0, cx - n);
        const int x1 = std::min(nx_ - 1, cx + n);
        const int y0 = std::max(0, cy - n);
        const int y1 = std::min(ny_ - 1, cy + n);
        const int z0 = std::max(0, cz - n);
        const int z1 = std::min(nz_ - 1, cz + n);

        for (int ix = x0; ix <= x1; ++ix) {
          const double px = x_min_ + ix * resolution_;
          const double dx = px - anchor.center[0];
          for (int iy = y0; iy <= y1; ++iy) {
            const double py = y_min_ + iy * resolution_;
            const double dy = py - anchor.center[1];
            for (int iz = z0; iz <= z1; ++iz) {
              const int linear = linearIndex(ix, iy, iz);
              const std::size_t ulinear =
                  static_cast<std::size_t>(linear);
              if (map.low[ulinear] == 0) {
                continue;
              }

              const double pz = z_min_ + iz * resolution_;
              const double dz = pz - anchor.center[2];
              const float clearance = static_cast<float>(
                  std::sqrt(dx * dx + dy * dy + dz * dz) -
                  static_cast<double>(anchor.radius));
              if (static_cast<double>(clearance) > proximity_margin_) {
                continue;
              }

              float& best = best_clearance_[ulinear];
              if (!std::isfinite(static_cast<double>(best))) {
                best = clearance;
                touched.push_back(linear);
              } else if (clearance < best) {
                best = clearance;
              }
            }
          }
        }
      }

      raw_total += touched.size();
      if (touched.empty()) {
        continue;
      }
      ++active_steps;

      std::sort(
          touched.begin(), touched.end(),
          [&](int a, int b) {
            const float ca =
                best_clearance_[static_cast<std::size_t>(a)];
            const float cb =
                best_clearance_[static_cast<std::size_t>(b)];
            if (ca != cb) {
              return ca < cb;
            }
            return map.confidence[static_cast<std::size_t>(a)] <
                   map.confidence[static_cast<std::size_t>(b)];
          });

      const std::size_t keep = std::min(
          touched.size(),
          static_cast<std::size_t>(max_pairs_per_step_));

      for (std::size_t i = 0; i < keep; ++i) {
        const int linear = touched[i];
        const std::size_t ulinear =
            static_cast<std::size_t>(linear);
        PairMeta pair;
        pair.point = pointForIndex(linear);
        pair.q = representative.q;
        pair.confidence = map.confidence[ulinear];
        pair.current_visibility =
            map.current_visibility[ulinear];
        pair.approx_body_clearance_m =
            best_clearance_[ulinear];
        pair.eval_timestep = representative.eval_timestep;
        pair.original_timestep = step;
        pairs.push_back(pair);
      }

      for (int linear : touched) {
        best_clearance_[static_cast<std::size_t>(linear)] =
            std::numeric_limits<float>::infinity();
      }
    }

    if (pairs.size() > static_cast<std::size_t>(max_pairs_)) {
      std::nth_element(
          pairs.begin(),
          pairs.begin() + max_pairs_,
          pairs.end(),
          [](const PairMeta& a, const PairMeta& b) {
            return a.approx_body_clearance_m <
                   b.approx_body_clearance_m;
          });
      pairs.resize(static_cast<std::size_t>(max_pairs_));
    }

    if (raw_pair_count) {
      *raw_pair_count = raw_total;
    }
    if (active_step_count) {
      *active_step_count = active_steps;
    }
    return pairs;
  }

  bool connectGpuSocket() {
    closeGpuSocket();

    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
      return false;
    }

    sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    if (gpu_socket_.size() >= sizeof(addr.sun_path)) {
      ::close(fd);
      ROS_ERROR_STREAM(
          "[C5.2h C++] GPU socket path too long: " << gpu_socket_);
      return false;
    }
    std::strncpy(
        addr.sun_path, gpu_socket_.c_str(), sizeof(addr.sun_path) - 1);

    if (::connect(
            fd,
            reinterpret_cast<sockaddr*>(&addr),
            sizeof(addr)) != 0) {
      ::close(fd);
      return false;
    }

    gpu_fd_ = fd;
    return true;
  }

  void closeGpuSocket() {
    if (gpu_fd_ >= 0) {
      ::close(gpu_fd_);
      gpu_fd_ = -1;
    }
  }

  bool queryGpu(
      const std::vector<float>& rows,
      std::size_t pair_count,
      std::vector<float>* distance,
      std::vector<float>* gradient,
      double* ipc_roundtrip_ms,
      ResponseHeader* response_header) {
    if (pair_count == 0 ||
        rows.size() != pair_count * 10) {
      return false;
    }

    if (gpu_fd_ < 0 && !connectGpuSocket()) {
      ROS_WARN_THROTTLE(
          1.0, "[C5.2h C++] GPU worker socket not connected");
      return false;
    }

    RequestHeader header;
    std::memcpy(header.magic, "CQ01", 4);
    header.pair_count = static_cast<uint32_t>(pair_count);

    const auto t0 = Clock::now();
    if (!sendAll(gpu_fd_, &header, sizeof(header)) ||
        !sendAll(
            gpu_fd_, rows.data(), rows.size() * sizeof(float))) {
      closeGpuSocket();
      return false;
    }

    ResponseHeader response;
    if (!recvAll(gpu_fd_, &response, sizeof(response))) {
      closeGpuSocket();
      return false;
    }

    if (std::memcmp(response.magic, "CR01", 4) != 0 ||
        response.pair_count != header.pair_count) {
      ROS_ERROR_THROTTLE(
          1.0, "[C5.2h C++] malformed GPU worker response");
      closeGpuSocket();
      return false;
    }

    distance->resize(pair_count);
    gradient->resize(pair_count * 7);
    if (!recvAll(
            gpu_fd_,
            distance->data(),
            distance->size() * sizeof(float)) ||
        !recvAll(
            gpu_fd_,
            gradient->data(),
            gradient->size() * sizeof(float))) {
      closeGpuSocket();
      return false;
    }

    if (ipc_roundtrip_ms) {
      *ipc_roundtrip_ms = msBetween(t0, Clock::now());
    }
    if (response_header) {
      *response_header = response;
    }
    return true;
  }

  std::vector<float> buildPairBuffer(
      const std::vector<PairMeta>& pairs) const {
    std::vector<float> rows;
    rows.resize(pairs.size() * 10);
    for (std::size_t i = 0; i < pairs.size(); ++i) {
      float* row = rows.data() + i * 10;
      row[0] = pairs[i].point[0];
      row[1] = pairs[i].point[1];
      row[2] = pairs[i].point[2];
      for (int j = 0; j < 7; ++j) {
        row[3 + j] = pairs[i].q[static_cast<std::size_t>(j)];
      }
    }
    return rows;
  }

  void appendJsonRecord(
      const ros::Time& anchor_stamp,
      const MapIndex& map,
      std::size_t anchor_count,
      std::size_t raw_pair_count,
      int active_step_count,
      const std::vector<PairMeta>& pairs,
      const std::vector<float>& distance,
      const std::vector<float>& gradient,
      double anchor_decode_ms,
      double selection_ms,
      double buffer_build_ms,
      double ipc_ms,
      const ResponseHeader& gpu,
      double pipeline_ms) {
    if (pairs.empty() || distance.empty()) {
      return;
    }

    const Stats d_stats = simpleStats(distance);
    std::vector<float> gradient_norm;
    gradient_norm.reserve(pairs.size());
    for (std::size_t i = 0; i < pairs.size(); ++i) {
      double norm_sq = 0.0;
      for (int j = 0; j < 7; ++j) {
        const double v = gradient[i * 7 + static_cast<std::size_t>(j)];
        norm_sq += v * v;
      }
      gradient_norm.push_back(static_cast<float>(std::sqrt(norm_sq)));
    }
    const Stats g_stats = simpleStats(gradient_norm);

    std::size_t negative = 0;
    std::size_t near_zero = 0;
    std::size_t positive = 0;
    std::size_t min_index = 0;
    std::unordered_map<int, float> per_step_min;
    std::unordered_map<int, int> per_step_count;

    for (std::size_t i = 0; i < distance.size(); ++i) {
      const double d = distance[i];
      if (d < -zero_band_) {
        ++negative;
      } else if (std::abs(d) <= zero_band_) {
        ++near_zero;
      } else {
        ++positive;
      }
      if (distance[i] < distance[min_index]) {
        min_index = i;
      }

      const int step = pairs[i].original_timestep;
      ++per_step_count[step];
      const auto it = per_step_min.find(step);
      if (it == per_step_min.end() || distance[i] < it->second) {
        per_step_min[step] = distance[i];
      }
    }

    std::vector<int> steps;
    steps.reserve(per_step_min.size());
    for (const auto& item : per_step_min) {
      steps.push_back(item.first);
    }
    std::sort(steps.begin(), steps.end());

    const PairMeta& min_pair = pairs[min_index];

    std::ofstream out(output_jsonl_, std::ios::out | std::ios::app);
    out << std::setprecision(9);
    out << "{";
    out << "\"record_index\":" << record_index_ << ",";
    out << "\"wall_time\":" << ros::WallTime::now().toSec() << ",";
    out << "\"anchor_cloud_stamp\":" << anchor_stamp.toSec() << ",";
    out << "\"map_cloud_stamp\":" << map.stamp.toSec() << ",";
    out << "\"anchor_count\":" << anchor_count << ",";
    out << "\"map_voxel_count\":" << map.total_voxel_count << ",";
    out << "\"low_confidence_voxel_count\":" << map.low_voxel_count << ",";
    out << "\"raw_local_pair_count\":" << raw_pair_count << ",";
    out << "\"retained_pair_count\":" << pairs.size() << ",";
    out << "\"active_step_count\":" << active_step_count << ",";

    out << "\"distance\":{";
    out << "\"min\":" << d_stats.min << ",";
    out << "\"mean\":" << d_stats.mean << ",";
    out << "\"max\":" << d_stats.max << "},";

    out << "\"gradient_norm\":{";
    out << "\"min\":" << g_stats.min << ",";
    out << "\"mean\":" << g_stats.mean << ",";
    out << "\"max\":" << g_stats.max << "},";

    out << "\"signed_counts\":{";
    out << "\"negative\":" << negative << ",";
    out << "\"near_zero\":" << near_zero << ",";
    out << "\"positive\":" << positive << ",";
    out << "\"negative_rate\":"
        << (static_cast<double>(negative) / distance.size()) << ",";
    out << "\"near_zero_rate\":"
        << (static_cast<double>(near_zero) / distance.size()) << ",";
    out << "\"positive_rate\":"
        << (static_cast<double>(positive) / distance.size()) << "},";

    out << "\"per_step_min_distance\":{";
    for (std::size_t i = 0; i < steps.size(); ++i) {
      if (i > 0) out << ",";
      const int step = steps[i];
      out << "\"" << step << "\":" << per_step_min[step];
    }
    out << "},";

    out << "\"per_step_pair_count\":{";
    for (std::size_t i = 0; i < steps.size(); ++i) {
      if (i > 0) out << ",";
      const int step = steps[i];
      out << "\"" << step << "\":" << per_step_count[step];
    }
    out << "},";

    out << "\"global_min_pair\":{";
    out << "\"point\":["
        << min_pair.point[0] << ","
        << min_pair.point[1] << ","
        << min_pair.point[2] << "],";
    out << "\"q\":[";
    for (int j = 0; j < 7; ++j) {
      if (j > 0) out << ",";
      out << min_pair.q[static_cast<std::size_t>(j)];
    }
    out << "],";
    out << "\"confidence\":" << min_pair.confidence << ",";
    out << "\"approx_body_clearance_m\":"
        << min_pair.approx_body_clearance_m << ",";
    out << "\"original_timestep\":"
        << min_pair.original_timestep << ",";
    out << "\"distance\":" << distance[min_index] << ",";
    out << "\"gradient\":[";
    for (int j = 0; j < 7; ++j) {
      if (j > 0) out << ",";
      out << gradient[min_index * 7 + static_cast<std::size_t>(j)];
    }
    out << "]},";

    out << "\"timing_ms\":{";
    out << "\"anchor_decode_ms\":" << anchor_decode_ms << ",";
    out << "\"pair_selection_ms\":" << selection_ms << ",";
    out << "\"pair_buffer_build_ms\":" << buffer_build_ms << ",";
    out << "\"ipc_roundtrip_ms\":" << ipc_ms << ",";
    out << "\"worker_h2d_ms\":" << gpu.h2d_ms << ",";
    out << "\"worker_inference_ms\":" << gpu.inference_ms << ",";
    out << "\"worker_d2h_ms\":" << gpu.d2h_ms << ",";
    out << "\"worker_total_ms\":" << gpu.worker_total_ms << ",";
    out << "\"map_index_build_ms\":" << map.build_ms << ",";
    out << "\"online_pipeline_ms\":" << pipeline_ms;
    out << "}";

    out << "}\n";
  }

  void publishConstraintBatch(
      const std_msgs::Header& source_header,
      const std::vector<PairMeta>& pairs,
      const std::vector<float>& distance,
      const std::vector<float>& gradient,
      double selection_ms,
      double buffer_build_ms,
      double ipc_ms,
      const ResponseHeader& gpu,
      double pipeline_ms) {
    if (pairs.empty() ||
        distance.size() != pairs.size() ||
        gradient.size() != pairs.size() * 7) {
      ROS_ERROR_THROTTLE(
          1.0, "[C5.3a C++] refusing malformed constraint batch");
      return;
    }

    care_collision_cdf::CollisionCDFConstraintBatch msg;
    msg.header = source_header;
    msg.num_pairs = static_cast<int32_t>(pairs.size());
    msg.dof = 7;
    msg.original_timestep.resize(pairs.size());
    msg.point_flat.resize(pairs.size() * 3);
    msg.q_linearization_flat.resize(pairs.size() * 7);
    msg.distance.resize(pairs.size());
    msg.gradient_flat.resize(pairs.size() * 7);

    for (std::size_t i = 0; i < pairs.size(); ++i) {
      msg.original_timestep[i] =
          static_cast<int32_t>(pairs[i].original_timestep);
      for (int j = 0; j < 3; ++j) {
        msg.point_flat[i * 3 + static_cast<std::size_t>(j)] =
            static_cast<double>(
                pairs[i].point[static_cast<std::size_t>(j)]);
      }
      for (int j = 0; j < 7; ++j) {
        msg.q_linearization_flat[
            i * 7 + static_cast<std::size_t>(j)] =
            static_cast<double>(
                pairs[i].q[static_cast<std::size_t>(j)]);
        msg.gradient_flat[
            i * 7 + static_cast<std::size_t>(j)] =
            static_cast<double>(
                gradient[i * 7 + static_cast<std::size_t>(j)]);
      }
      msg.distance[i] = static_cast<double>(distance[i]);
    }

    msg.pair_selection_ms = selection_ms;
    msg.pair_buffer_build_ms = buffer_build_ms;
    msg.ipc_roundtrip_ms = ipc_ms;
    msg.gpu_h2d_ms = gpu.h2d_ms;
    msg.gpu_inference_ms = gpu.inference_ms;
    msg.gpu_d2h_ms = gpu.d2h_ms;
    msg.gpu_worker_total_ms = gpu.worker_total_ms;
    msg.online_pipeline_ms = pipeline_ms;

    constraint_batch_pub_.publish(msg);
  }

  void publishSummary(
      std::size_t pair_count,
      int active_step_count,
      double d_min,
      double negative_rate,
      double anchor_decode_ms,
      double selection_ms,
      double buffer_build_ms,
      double ipc_ms,
      const ResponseHeader& gpu,
      double pipeline_ms,
      double map_index_ms) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3)
        << "C5_2H_CPP_GPU "
        << "pairs=" << pair_count
        << " steps=" << active_step_count
        << " d_min=" << d_min
        << " neg_rate=" << negative_rate
        << " anchor_decode_ms=" << anchor_decode_ms
        << " selection_ms=" << selection_ms
        << " buffer_ms=" << buffer_build_ms
        << " ipc_ms=" << ipc_ms
        << " gpu_h2d_ms=" << gpu.h2d_ms
        << " gpu_infer_ms=" << gpu.inference_ms
        << " gpu_d2h_ms=" << gpu.d2h_ms
        << " pipeline_ms=" << pipeline_ms
        << " map_index_ms=" << map_index_ms;

    std_msgs::String msg;
    msg.data = oss.str();
    summary_pub_.publish(msg);
  }

  void timerCallback(const ros::TimerEvent&) {
    const auto pipeline_t0 = Clock::now();

    sensor_msgs::PointCloud2ConstPtr anchor_cloud;
    ros::Time anchor_received;
    std::shared_ptr<MapIndex> map;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      anchor_cloud = latest_anchor_cloud_;
      anchor_received = latest_anchor_received_;
      map = latest_map_;
    }

    if (!anchor_cloud || !map) {
      return;
    }

    const ros::Time now = ros::Time::now();
    if ((now - anchor_received).toSec() > anchor_stale_s_ ||
        (now - map->received).toSec() > map_stale_s_) {
      return;
    }

    if (anchor_cloud->header.stamp == last_processed_anchor_stamp_) {
      return;
    }
    last_processed_anchor_stamp_ = anchor_cloud->header.stamp;

    const auto decode_t0 = Clock::now();
    std::vector<Anchor> anchors = decodeAnchors(*anchor_cloud);
    const auto decode_t1 = Clock::now();
    if (anchors.empty()) {
      return;
    }

    std::size_t raw_pair_count = 0;
    int active_step_count = 0;
    const auto selection_t0 = Clock::now();
    std::vector<PairMeta> pairs = buildPairs(
        anchors, *map, &raw_pair_count, &active_step_count);
    const auto selection_t1 = Clock::now();
    if (pairs.empty()) {
      return;
    }

    const auto buffer_t0 = Clock::now();
    std::vector<float> rows = buildPairBuffer(pairs);
    const auto buffer_t1 = Clock::now();

    std::vector<float> distance;
    std::vector<float> gradient;
    double ipc_ms = 0.0;
    ResponseHeader gpu{};
    if (!queryGpu(
            rows,
            pairs.size(),
            &distance,
            &gradient,
            &ipc_ms,
            &gpu)) {
      ROS_WARN_THROTTLE(
          1.0, "[C5.2h C++] GPU query failed");
      return;
    }

    const double pipeline_ms =
        msBetween(pipeline_t0, Clock::now());
    const double anchor_decode_ms =
        msBetween(decode_t0, decode_t1);
    const double selection_ms =
        msBetween(selection_t0, selection_t1);
    const double buffer_ms =
        msBetween(buffer_t0, buffer_t1);

    const Stats d_stats = simpleStats(distance);
    std::size_t negative = 0;
    for (float d : distance) {
      if (static_cast<double>(d) < -zero_band_) {
        ++negative;
      }
    }
    const double negative_rate =
        static_cast<double>(negative) /
        static_cast<double>(distance.size());

    publishConstraintBatch(
        anchor_cloud->header,
        pairs,
        distance,
        gradient,
        selection_ms,
        buffer_ms,
        ipc_ms,
        gpu,
        pipeline_ms);

    appendJsonRecord(
        anchor_cloud->header.stamp,
        *map,
        anchors.size(),
        raw_pair_count,
        active_step_count,
        pairs,
        distance,
        gradient,
        anchor_decode_ms,
        selection_ms,
        buffer_ms,
        ipc_ms,
        gpu,
        pipeline_ms);

    publishSummary(
        pairs.size(),
        active_step_count,
        d_stats.min,
        negative_rate,
        anchor_decode_ms,
        selection_ms,
        buffer_ms,
        ipc_ms,
        gpu,
        pipeline_ms,
        map->build_ms);

    ROS_INFO_STREAM_THROTTLE(
        1.0,
        "[C5.2h C++] pairs=" << pairs.size()
        << " pipeline=" << pipeline_ms << " ms"
        << " selection=" << selection_ms << " ms"
        << " ipc=" << ipc_ms << " ms"
        << " gpu=" << gpu.inference_ms << " ms"
        << " dmin=" << d_stats.min);

    ++record_index_;
  }

 private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  std::string anchor_topic_;
  std::string map_topic_;
  std::string summary_topic_;
  std::string constraint_batch_topic_;
  std::string output_jsonl_;
  std::string gpu_socket_;

  double rate_hz_ = 20.0;
  double anchor_stale_s_ = 0.25;
  double map_stale_s_ = 0.50;
  double confidence_threshold_ = 0.50;
  double proximity_margin_ = 0.075;
  int max_pairs_per_step_ = 250;
  int max_pairs_ = 8000;
  double zero_band_ = 0.05;

  double x_min_ = -0.95;
  double x_max_ = 0.95;
  double y_min_ = -0.95;
  double y_max_ = 0.95;
  double z_min_ = 0.0;
  double z_max_ = 1.15;
  double resolution_ = 0.05;
  int nx_ = 0;
  int ny_ = 0;
  int nz_ = 0;
  std::size_t grid_size_ = 0;

  ros::Subscriber anchor_sub_;
  ros::Subscriber map_sub_;
  ros::Publisher summary_pub_;
  ros::Publisher constraint_batch_pub_;
  ros::Timer timer_;

  mutable std::mutex mutex_;
  sensor_msgs::PointCloud2ConstPtr latest_anchor_cloud_;
  ros::Time latest_anchor_received_;
  std::shared_ptr<MapIndex> latest_map_;
  ros::Time last_processed_anchor_stamp_;

  std::vector<float> best_clearance_;

  int gpu_fd_ = -1;
  std::size_t record_index_ = 0;
};

}  // namespace care_collision_cdf

int main(int argc, char** argv) {
  ros::init(argc, argv, "cpp_forbidden_voxel_gpu_shadow");
  try {
    care_collision_cdf::CppForbiddenVoxelGpuShadow node;
    ros::spin();
  } catch (const std::exception& e) {
    ROS_FATAL_STREAM("[C5.2h C++] fatal: " << e.what());
    return 1;
  }
  return 0;
}
